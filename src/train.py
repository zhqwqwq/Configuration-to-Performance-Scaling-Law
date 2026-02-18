from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import torch
import json
import time
import os

from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
import torch.distributed as dist
from torch.amp import autocast

from utils.load_config import load_config
from model import ScalingLawForecaster

cfg = load_config()

# DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=cfg['ddp']['backend'])
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    ddp_world_size = dist.get_world_size()
    print(f"DDP initialized: rank {ddp_rank}, world size {ddp_world_size}, device {device}")
else:
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda'


# random seed.
def seed_everything(base_seed: int, rank: int):
    seed = base_seed + rank
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
seed_everything(cfg.seed, ddp_rank)

def set_base_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("base.") or ".base." in name:
            param.requires_grad = requires_grad

def get_trainable_params(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        get_trainable_params(model),
        lr=lr,
        weight_decay=cfg.training.weight_decay,
        betas=[cfg.training.beta1, cfg.training.beta2],
    )

def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    if total_steps <= 0:
        return None
    warmup_steps = max(0, min(warmup_steps, total_steps))
    if warmup_steps == 0:
        lr_lambda = lambda step: 1.0 - (step / total_steps)
    else:
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 - step / total_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def compute_warmup_steps_ratio(total_steps: int, warmup_ratio: float) -> int:
    total_steps = int(total_steps)
    if total_steps <= 0:
        return 0
    if warmup_ratio is None:
        return 0
    warmup_ratio = float(warmup_ratio)
    if warmup_ratio <= 0:
        return 0
    return max(1, min(int(total_steps * warmup_ratio), total_steps))


# model
model = ScalingLawForecaster(
    base_model_name=cfg.model.base_model,
).to(device)
model = model.float()  # Convert model to FP32

if ddp:
    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        # mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # allow toggling requires_grad between stages
    )


# loss
def pointwise_loss(pred, target, loss_type="mse", huber_delta=1.0):
    err = (pred - target).to(torch.float32)
    if loss_type == "mse":
        return err ** 2
    elif loss_type == "mae":
        return err.abs()
    elif loss_type == "huber":
        abs_err = err.abs()
        quad = 0.5 * (err ** 2)
        lin = huber_delta * (abs_err - 0.5 * huber_delta)
        return torch.where(abs_err <= huber_delta, quad, lin)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


if cfg.training.amp:
    raise NotImplementedError("This script is for fp32.")


# dataset and dataloader
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.examples = torch.load(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        number_type_mask = torch.tensor(ex['number_type_mask'], dtype=torch.long)
        loss_mask = torch.tensor(ex['loss_mask'], dtype=torch.bool) if 'loss_mask' in ex else number_type_mask != 0
        return (
            torch.tensor(ex['input_ids'], dtype=torch.long),
            torch.tensor(ex['is_number_mask'], dtype=torch.bool),
            torch.tensor(ex['number_values_filled'], dtype=torch.float),
            number_type_mask,
            loss_mask
        )

def collate_fn(batch):
    input_ids, masks, values, number_types, loss_mask = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
    values = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0.0)
    number_types = torch.nn.utils.rnn.pad_sequence(number_types, batch_first=True, padding_value=0)
    loss_mask = torch.nn.utils.rnn.pad_sequence(loss_mask, batch_first=True, padding_value=False)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, masks, values, number_types, loss_mask

train_ds = TokenizedDataset(cfg.data.dataset_path + "/train_tok.pt")
train_sampler = DistributedSampler(train_ds, num_replicas=ddp_world_size, rank=ddp_rank)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn, sampler=train_sampler, pin_memory=True)

if cfg.logging.log == True:
    in_val_ds = TokenizedDataset(cfg.data.dataset_path + "/in_val_tok.pt")
    in_val_sampler = DistributedSampler(in_val_ds, num_replicas=ddp_world_size, rank=ddp_rank)
    in_val_loader = torch.utils.data.DataLoader(in_val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn, sampler=in_val_sampler, pin_memory=True)

    ood_val_ds = TokenizedDataset(cfg.data.dataset_path + "/ood_val_tok.pt")
    ood_val_sampler = DistributedSampler(ood_val_ds, num_replicas=ddp_world_size, rank=ddp_rank)
    ood_val_loader = torch.utils.data.DataLoader(ood_val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn, sampler=ood_val_sampler, pin_memory=True)

if master_process:
    print(f"Number of samples in train_ds: {len(train_ds)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")

# Read dataset sub-config
_ds_cfg = OmegaConf.load(cfg.data.dataset_path + "/args.yaml")
if "dataset" not in cfg:
    cfg.dataset = OmegaConf.create()
cfg.dataset = OmegaConf.merge(cfg.dataset, _ds_cfg)


# hyperparameters
grad_clip = cfg.training.grad_clip
stage1_cfg = cfg.training.stage1
stage2_cfg = cfg.training.stage2
stage1_epochs = int(stage1_cfg.epochs)
stage2_epochs = int(stage2_cfg.epochs)
stage1_lr = float(stage1_cfg.lr)
stage2_lr = float(stage2_cfg.lr)

name = (
    f"{cfg.output_prefix + '_' if cfg.output_prefix else ''}"
    f"@{cfg.data.dataset_path.split('/')[-2] + '_' + cfg.data.dataset_path.split('/')[-1]}"
    f"s1ep{stage1_epochs}_s2ep{stage2_epochs}_s1lr{stage1_lr}_s2lr{stage2_lr}_"
    f"wd{cfg.training.weight_decay}_bs{cfg.training.batch_size * ddp_world_size}_"
    f"rs{cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
)

if master_process:
    print(cfg)
    print(name)

steps_per_epoch = len(train_loader)

# Training loop
iter_num = 0
best_val_loss = float('inf')
best_ood_val_loss = float('inf')

log_dir = f"{cfg.logging.log_dir}/{name}"
log_path = os.path.join(log_dir, 'train.log')
checkpoint_dir = os.path.join(log_dir, 'checkpoints')
if master_process:
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if cfg.model_path_out != None:
        with open(cfg.model_path_out, "w") as f:
            f.write(log_dir)
            f.flush()

    run = wandb.init(project=cfg.logging.wandb_project, name=name, config=OmegaConf.to_container(cfg, resolve=True))
    wandb.watch(model, log='all')

    cfg.logging.wandb_run_id = run.id
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

stages = [
    ("stage1", stage1_cfg, True),
    ("stage2", stage2_cfg, False),
]

global_epoch = 0

for stage_name, stage_cfg, freeze_base in stages:
    stage_epochs = int(stage_cfg.epochs)
    if stage_epochs <= 0:
        if master_process:
            print(f"Skipping {stage_name} because epochs={stage_epochs}")
        continue

    set_base_requires_grad(model, not freeze_base)
    stage_lr = float(stage_cfg.lr)
    optimizer = build_optimizer(model, stage_lr)

    stage_total_steps = steps_per_epoch * stage_epochs
    if stage_name == "stage1":
        warmup_steps = compute_warmup_steps_ratio(
            stage_total_steps,
            stage_cfg.get("warmup_ratio", 0.1),
        )
    else:
        warmup_steps = int(stage_cfg.get("warmup_steps", 0))
    scheduler = build_scheduler(optimizer, stage_total_steps, warmup_steps)
    trainable_params = get_trainable_params(model)

    if master_process:
        print(f"=== {stage_name} ===")
        print(f"epochs={stage_epochs}, lr={stage_lr}, warmup_steps={warmup_steps}, trainable_params={len(trainable_params)}")

    for epoch in range(stage_epochs):
        model.train()
        train_sampler.set_epoch(global_epoch)
        running_loss = 0.0
        running_mse_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{stage_epochs}", dynamic_ncols=True, disable=not master_process)
        for ids, attn, mask, vals, number_types, loss_mask in progress_bar:
            ids, attn, mask, vals, number_types, loss_mask = ids.to(device, non_blocking=True), attn.to(device, non_blocking=True), mask.to(device, non_blocking=True), vals.to(device, non_blocking=True, dtype=torch.float32), number_types.to(device, non_blocking=True, dtype=torch.long), loss_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=False):  # FP32 training
                pred = model(ids, mask, vals, attention_mask=attn)
                pred = pred[:, :-1]
                target = vals[:, 1:]
                mask_shifted = (loss_mask)[:, 1:]
                m = mask_shifted.float()
                loss_per_pos = pointwise_loss(pred, target, cfg.training.loss_type, getattr(cfg.training, "huber_delta", 1.0))
                denom = m.sum().clamp(min=1)
                loss = (loss_per_pos * m).sum() / denom
                if cfg.training.loss_type == "mse":
                    mse_loss = loss
                else:
                    mse_loss_per_loss = pointwise_loss(pred, target, "mse", getattr(cfg.training, "huber_delta", 1.0))
                    mse_loss = (mse_loss_per_loss * m).sum() / m.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)  # gradient clipping
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            running_mse_loss += mse_loss.item()

            # logging
            if iter_num % cfg.logging.log_steps == 0 and master_process:
                with torch.no_grad():
                    if iter_num == 0:
                        avg_loss = running_loss
                        avg_mse_loss = running_mse_loss
                    else:
                        avg_loss = running_loss / cfg.logging.log_steps
                        avg_mse_loss = running_mse_loss / cfg.logging.log_steps
                    print(f"Epoch {global_epoch}, Iter {iter_num}, Loss: {avg_loss:.4f}")
                    wandb.log({"train/loss": avg_loss}, step=iter_num)
                    wandb.log({"train/mse_loss": avg_mse_loss}, step=iter_num)

                    wandb.log({
                        "train/max_pred": pred.max().item(),
                        "train/max_target": target.max().item()
                    }, step=iter_num)
                    running_loss = 0.0
                    running_mse_loss = 0.0

            if iter_num % cfg.logging.eval_steps == 0 and cfg.logging.log:
                model.eval()
                
                # Store validation results to decide on checkpointing after the loop
                val_results = {}
                
                for (vname, loader) in [("in_val", in_val_loader),("ood_val", ood_val_loader)]:
                    val_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                    val_mse_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                    val_count_sum = torch.zeros((), device=device, dtype=torch.float32)
                    val_type_loss_sum = torch.zeros(5, device=device, dtype=torch.float32)
                    val_type_count_sum = torch.zeros(5, device=device, dtype=torch.float32)

                    with torch.no_grad():
                        progress_bar_val = tqdm(loader, desc=f"Epoch {global_epoch}",
                                                dynamic_ncols=True, disable=not master_process)

                        for ids, attn, mask, vals, number_types, loss_mask in progress_bar_val:
                            ids = ids.to(device, non_blocking=True)
                            attn = attn.to(device, non_blocking=True)
                            mask = mask.to(device, non_blocking=True)
                            vals = vals.to(device, non_blocking=True, dtype=torch.float32)
                            number_types = number_types.to(device, non_blocking=True, dtype=torch.long)
                            loss_mask = loss_mask.to(device, non_blocking=True)

                            with autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16, enabled=False):
                                pred = model(ids, mask, vals, attention_mask=attn)
                                pred_shifted = pred[:, :-1]
                                target_shifted = vals[:, 1:]
                                mask_shifted = loss_mask[:, 1:]
                                number_types_shifted = number_types[:, 1:]

                            m = mask_shifted.float()
                            loss_per_pos = pointwise_loss(pred_shifted.float(), target_shifted.float(),
                                                    cfg.training.loss_type, getattr(cfg.training, "huber_delta", 1.0)) * m
                            if cfg.training.loss_type == "mse":
                                mse_loss_per_pos = loss_per_pos
                            else:
                                mse_loss_per_pos = pointwise_loss(pred_shifted.float(), target_shifted.float(),
                                                    "mse", huber_delta=None) * m

                            val_loss_sum += loss_per_pos.sum()
                            val_mse_loss_sum += mse_loss_per_pos.sum()
                            val_count_sum += m.sum()

                            for idx in range(1, 3):
                                type_m = number_types_shifted == idx
                                type_loss = loss_per_pos[type_m].sum()
                                type_count = m[type_m].sum()
                                val_type_loss_sum[idx] += type_loss
                                val_type_count_sum[idx] += type_count


                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(val_mse_loss_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(val_count_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(val_type_count_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(val_type_loss_sum, op=dist.ReduceOp.SUM)

                    val_loss = (val_loss_sum / val_count_sum).item() if val_count_sum.item() > 0 else float('nan')
                    val_mse_loss = (val_mse_loss_sum / val_count_sum).item() if val_count_sum.item() > 0 else float('nan')
                    
                    val_results[vname] = {
                        'val_loss': val_loss,
                        'val_mse_loss': val_mse_loss,
                        'val_type_loss_sum': val_type_loss_sum.clone(),
                        'val_type_count_sum': val_type_count_sum.clone(),
                    }

                    if master_process:
                        print(f"{vname} loss: {val_loss:.6f}")
                        print(f"{vname} mse loss: {val_mse_loss:.6f}")
                        wandb.log({f"{vname}/loss": val_loss, f"{vname}/mse_loss": val_mse_loss}, step=iter_num)

                        for (idx, type_name) in [(1, "tls"), (2, "eval_ls"), (3, "lm_eval_bpb"), (4, "lm_eval_acc")]:
                            type_loss = val_type_loss_sum[idx] / torch.clamp_min(val_type_count_sum[idx], 1).item()
                            print(f"{vname} {type_name} loss: {type_loss:.6f}")
                            wandb.log({f"{vname}/{type_name} loss": type_loss}, step=iter_num)

                # Check if we need to save checkpoints (only gather state dict once if needed)
                need_save_in_val = val_results["in_val"]['val_loss'] < best_val_loss
                need_save_ood_val = val_results["ood_val"]['val_loss'] < best_ood_val_loss
                
                # Update best losses on ALL ranks so the save decision is consistent
                if need_save_in_val:
                    best_val_loss = val_results["in_val"]['val_loss']
                if need_save_ood_val:
                    best_ood_val_loss = val_results["ood_val"]['val_loss']

                if need_save_in_val or need_save_ood_val:
                    # Synchronize before gathering state dict
                    if dist.is_available() and dist.is_initialized():
                        torch.cuda.synchronize()
                        dist.barrier()
                    
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    ):
                        state_dict = model.state_dict()

                    # Synchronize after gathering state dict
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                        torch.cuda.synchronize()

                    if master_process:
                        if need_save_in_val:
                            checkpoint = {
                                'epoch': global_epoch,
                                'model': state_dict,
                                'iter_num': iter_num,
                                'val_loss': best_val_loss
                            }
                            print("Saving to:", f"{checkpoint_dir}/checkpoint_min_val_loss.pt")
                            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_min_val_loss.pt")
                            print("Saved best in_val model checkpoint.")

                        if need_save_ood_val:
                            checkpoint = {
                                'epoch': global_epoch,
                                'model': state_dict,
                                'iter_num': iter_num,
                                'val_loss': best_ood_val_loss
                            }
                            print("Saving to:", f"{checkpoint_dir}/checkpoint_min_ood_val_loss.pt")
                            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_min_ood_val_loss.pt")
                            print("Saved best ood_val model checkpoint.")

                # Barrier before switching back to train mode
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                model.train()

            if iter_num % cfg.logging.save_steps == 0 and iter_num != 0 and cfg.logging.log_intermediate_ckpt:
                # Synchronize before gathering state dict
                if dist.is_available() and dist.is_initialized():
                    torch.cuda.synchronize()
                    dist.barrier()

                with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    ):
                    state_dict = model.state_dict()

                # Synchronize after gathering state dict
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                    torch.cuda.synchronize()

                if master_process:
                    checkpoint = {
                        'epoch': global_epoch,
                        'model': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'iter_num': iter_num,
                    }
                    torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_step{iter_num}.pt")

            iter_num += 1

        global_epoch += 1

    if stage_name == "stage1":
        # Synchronize before gathering state dict
        if dist.is_available() and dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        # Synchronize after gathering state dict
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            torch.cuda.synchronize()

        if master_process:
            checkpoint = {
                'epoch': global_epoch - 1,
                'stage': stage_name,
                'model': state_dict,
                'iter_num': iter_num,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_stage1_final.pt")
            print("Saved stage1 checkpoint.")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


wandb.finish()

# Synchronize before gathering final state dict
if dist.is_available() and dist.is_initialized():
    torch.cuda.synchronize()
    dist.barrier()

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    state_dict = model.state_dict()

# Synchronize after gathering final state dict
if dist.is_available() and dist.is_initialized():
    dist.barrier()
    torch.cuda.synchronize()

if master_process:
    checkpoint = {
        'epoch': global_epoch - 1,
        'model': state_dict,
        'iter_num': iter_num,
    }
    torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_final.pt")
