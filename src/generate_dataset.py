from pathlib import Path
from tqdm import tqdm
import random
import torch
import json
import os
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from utils.load_config import load_config
from utils.data import recover_text


chinchilla_params = {
    "marin": {
        'a': 3.997941824442712, 
        'alpha': 0.44174138432888976, 
        'b': 0.8251456329446542, 
        'beta': 0.1676451054204669, 
        'c': 2.2814165287450208
    },
    "steplaw": {
        'a': 1.85240120191131, 
        'alpha': 0.12576480455477607, 
        'b': 0.7308261081070557, 
        'beta': 0.3337389553968635, 
        'c': 1.2122888212135527
    }
}

def chinchilla(params, model_size, data_size):
    return params['a'] * (model_size ** (-params['alpha'])) + params['b'] * (data_size ** (-params['beta'])) + params['c']

def prepare_nl(meta_data):
    nl = []
    for k, v in meta_data.items():
        if k == "model size":
            nl.append(f"model size: {int(v)}")
        elif k == "data size":
            nl.append(f"data size: {v:.2f}")
        else:
            nl.append(f"{k}: {v}")
    return nl

def tokenize_example(example: dict, tokenizer, args): 
    """
    Builds:
      - input_ids
      - is_number_mask         (1 only for the single numeric placeholder token)
      - number_type_mask       (0=hyperparameters, 1=final_loss, 2=intermediate_loss)
      - number_values_filled   (the numeric value; 0 for non-number tokens)
      - number_types           (string tag for debugging/analysis)
    """
    
    tokenized_examples = []

    input_ids = []
    is_number_mask = []
    number_type_mask = []
    number_values_filled = []
    number_types = []
    loss_mask = []

    NUM_TOKEN_ID = 0

    # --- helpers ---
    def emit_text(text: str):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        L = len(ids)
        input_ids.extend(ids)
        is_number_mask.extend([0] * L)
        number_type_mask.extend([0] * L)      # non-number tokens don't train any numeric head
        number_values_filled.extend([0.0] * L)
        number_types.extend([None] * L)
        loss_mask.extend([False] * L)

    def emit_step(step_val):
        # keep consistent formatting
        ids = tokenizer(str(step_val), add_special_tokens=False).input_ids
        L = len(ids)
        input_ids.extend(ids)
        is_number_mask.extend([0] * L)
        number_type_mask.extend([0] * L)
        number_values_filled.extend([0.0] * L)
        number_types.extend([None] * L)
        loss_mask.extend([False] * L)

    def emit_number(value: float, num_mask_code: int, num_type_str: str, compute_loss: int):
        # single numeric placeholder token
        input_ids.append(NUM_TOKEN_ID)
        is_number_mask.append(1)
        number_type_mask.append(num_mask_code)
        number_values_filled.append(float(value))
        number_types.append(num_type_str)
        loss_mask.append(compute_loss)

    source = example["source"]
    name = example["name"]
    meta_data = example["conf"]

    # --- optional NL context ---
    if args.context == True:
        if args.number_token_input == False:
            nl_section = prepare_nl(meta_data)
            emit_text('\n'.join(nl_section) + "\n")
        else : # Additional take num layers, num heads hidden dim and max_grad_norm into account.
            for k, v in meta_data.items():
                if k == "data size":
                    emit_text("data size: ")
                    emit_number(v, num_mask_code=0, num_type_str="data size", compute_loss=0) # v ~ [1, 100]
                    emit_text("\n")
                elif k == "model size":
                    emit_text("model size: ")
                    emit_number(v / 100, num_mask_code=0, num_type_str="model size", compute_loss=0) # v / 100 ~ [1, 10]
                    emit_text("\n")
                elif k == "num layers":
                    emit_text("num layers: ")
                    emit_number(v, num_mask_code=0, num_type_str="num layers", compute_loss=0) # v ~ [1, 10]
                    emit_text("\n")
                elif k == "num heads":
                    emit_text("num heads: ")
                    emit_number(v, num_mask_code=0, num_type_str="num heads", compute_loss=0) # v ~ [1, 10]
                    emit_text("\n")
                elif k == "hidden dim":
                    emit_text("hidden dim: ")
                    emit_number(v / 100, num_mask_code=0, num_type_str="hidden dim", compute_loss=0) # v / 100 ~ [1, 100]
                    emit_text("\n")
                elif k == "learning rate":
                    emit_text("learning rate: ")
                    emit_number(v * 1e4, num_mask_code=0, num_type_str="learning rate", compute_loss=0) # v * 1e4 ~ [1, 20]
                    emit_text("\n")
                elif k == "warmup":
                    emit_text("warmup: ")
                    # emit_number(v * 100, num_mask_code=0, num_type_str="warmup", compute_loss=0) # v / 1e3 ~ [1, 100]
                    emit_number(v / 100, num_mask_code=0, num_type_str="warmup", compute_loss=0) # v / 100 ~ [1, 40] # changed to exact step, instead of ratio.
                    emit_text("\n")
                elif k == "weight decay":
                    emit_text("weight decay: ")
                    emit_number(v * 100, num_mask_code=0, num_type_str="weight decay", compute_loss=0) # v * 1e2 ~ [1, 100]
                    emit_text("\n")
                elif k == "batch size":
                    emit_text("batch size: ")
                    emit_number(v / 10, num_mask_code=0, num_type_str="batch size", compute_loss=0) # v / 10 ~ [1, 20]
                    emit_text("\n")
                elif k == "max_grad_norm":
                    emit_text("max_grad_norm: ")
                    emit_number(v, num_mask_code=0, num_type_str="max_grad_norm", compute_loss=0) # v * 10 ~ [1, 2]
                    emit_text("\n")
                elif k == "min_lr":
                    emit_text("minlr: ")
                    emit_number(v * 1e4, num_mask_code=0, num_type_str="minlr", compute_loss=0) # v * 1e4 ~ [1, 20]
                    emit_text("\n")
                elif k == "min_lr_ratio":
                    emit_text("minlr ratio: ")
                    emit_number(v * 200, num_mask_code=0, num_type_str="minlr_ratio", compute_loss=0) # v * 200 ~ [1, 100]
                    emit_text("\n")
                elif k == "adam_lr":
                    emit_text("adam_lr: ")
                    emit_number(v * 1e4, num_mask_code=0, num_type_str="adam_lr", compute_loss=0) # v * 1e4 ~ [1, 20]
                    emit_text("\n")
                elif k == "block_size":
                    emit_text("block_size: ")
                    emit_number(v / 50, num_mask_code=0, num_type_str="block_size", compute_loss=0) # v / 100 ~ [2.56, 10.24 ]
                    emit_text("\n")
                else:
                    emit_text(f"{k}: {v}"+"\n")
        
    max_step = example["max_step"]
    
    ms = meta_data["model size"]
    ds = meta_data["data size"]

    if source == "steplaw": params = chinchilla_params["steplaw"]
    elif source == "marin": params = chinchilla_params["marin"]
    else: raise NotImplementedError
    chinchilla_loss = chinchilla(params, ms, ds)

    if source == "marin":
        intermediate_steps = example["c4en_steps"]
        intermediate_loss = example["c4en_losses"]
    elif source == "steplaw":
        intermediate_steps = example["smoothed_loss_steps"]
        intermediate_loss = example["smoothed_loss"]
    else: raise NotImplementedError
    if len(intermediate_steps) > 15: # No more than 30 intermediate points.
        stride = len(intermediate_steps) // 15
        intermediate_steps = intermediate_steps[::stride]
        intermediate_loss = intermediate_loss[::stride]
    
    emit_text("max step: ")

    if args.number_token_input == False:
        emit_step(max_step)
    else:
        emit_number(max_step / 1e3, num_mask_code=0, num_type_str="step", compute_loss=0) # step / 1e3 ~ [1, 100]
    
    # intermediate loss.
    if args.intermediate == True:
        for step, step_loss in zip(intermediate_steps, intermediate_loss):
            input_ids_cur = input_ids.copy()
            is_number_mask_cur = is_number_mask.copy()
            number_type_mask_cur = number_type_mask.copy()
            number_values_filled_cur = number_values_filled.copy()
            number_types_cur = number_types.copy()
            loss_mask_cur = loss_mask.copy()

            def emit_text_cur(text: str):
                ids = tokenizer(text, add_special_tokens=False).input_ids
                L = len(ids)
                input_ids_cur.extend(ids)
                is_number_mask_cur.extend([0] * L)
                number_type_mask_cur.extend([0] * L)
                number_values_filled_cur.extend([0.0] * L)
                number_types_cur.extend([None] * L)
                loss_mask_cur.extend([False] * L)

            def emit_number_cur(value: float, num_mask_code: int, num_type_str: str, compute_loss: int):
                input_ids_cur.append(NUM_TOKEN_ID)
                is_number_mask_cur.append(1)
                number_type_mask_cur.append(num_mask_code)
                number_values_filled_cur.append(float(value))
                number_types_cur.append(num_type_str)
                loss_mask_cur.append(bool(compute_loss))

            emit_text_cur("frac: ")
            emit_number_cur(step / max_step, num_mask_code=0, num_type_str="frac", compute_loss=0)

            if args.rescale == True:
                recorded_loss = step_loss - chinchilla_loss
            else:
                recorded_loss = step_loss

            emit_text_cur("loss: ")
            emit_number_cur(recorded_loss, num_mask_code=2, num_type_str="intermediate_train_loss", compute_loss=1)

            result = {
                "input_ids": input_ids_cur,
                "is_number_mask": is_number_mask_cur,
                "number_type_mask": number_type_mask_cur,
                "number_values_filled": number_values_filled_cur,
                "number_types": number_types_cur,
                "loss_mask": loss_mask_cur,
                "max_step": max_step,
                "frac": step / max_step,
                "name": name,
                "meta_data": meta_data,
                "step_loss": step_loss,
                "chinchilla": chinchilla_loss,
                "task": "intermediate_loss",
            }
            tokenized_examples.append(result)

    # final loss.
    emit_text("final loss: ")
    final_loss = example["final_loss"]
    if args.rescale == True:
        recorded_loss = final_loss - chinchilla_loss
    else:
        recorded_loss = final_loss
    emit_number(recorded_loss, num_mask_code=1, num_type_str="train_loss", compute_loss=1)

    result = {
        "input_ids": input_ids,
        "is_number_mask": is_number_mask,
        "number_type_mask": number_type_mask,
        "number_values_filled": number_values_filled,
        "number_types": number_types,
        "loss_mask": loss_mask,
        "max_step": max_step,
        "name": name,
        "meta_data": meta_data,
        "loss": final_loss,
        "chinchilla": chinchilla_loss,
        "task": "final_loss",
    }
    tokenized_examples.append(result)

    # print(recover_text(input_ids, is_number_mask, number_values_filled, tokenizer))
    # print("sum of loss_mask", sum(loss_mask))
    # print("sum of number tokens", sum(is_number_mask))
    # print("sum of train loss", sum([(item == 1) for item in number_type_mask]))
    # print("chinchilla loss", chinchilla_loss)
    # breakpoint()

    return tokenized_examples

def main():
    args = load_config()
    print(args)
    for k, v in vars(args).items():
        print(k, type(v))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer_name = "qwen" if "Qwen" in args.tokenizer_name else "smol"
    
    # Get run_names
    dataset_splits = {"train": {}, "in_val": {}, "ood_val": {}}
    
    # prepare keys
    for ds in args.data_source:
        if ds == "marin":
            key_path = "processed_data/keys/marin_keys_model_split.json"
            with open(key_path, "r") as f:
                marin_examples_list = json.load(f)
            dataset_splits["train"][ds] = marin_examples_list["train"]
            dataset_splits["in_val"][ds] = marin_examples_list["in_val"]
            dataset_splits["ood_val"][ds] = marin_examples_list["ood_val"]
        elif ds == "steplaw":
            key_path = "processed_data/keys/steplaw_keys_model_split.json"
            with open(key_path, "r") as f:
                steplaw_examples_list = json.load(f)
            dataset_splits["train"][ds] = steplaw_examples_list["train"]
            dataset_splits["in_val"][ds] = steplaw_examples_list["in_val"]
            dataset_splits["ood_val"][ds] = steplaw_examples_list["ood_val"]
        else: raise NotImplementedError
    
    # load processed_data
    processed_data = {}
    with open ("processed_data/marin_processed_data.json", "r") as f:
        processed_data_marin = json.load(f)
        processed_data["marin"] = {ent["name"]: ent for ent in processed_data_marin}
    with open ("processed_data/steplaw_processed_data.json", "r") as f:
        processed_data_steplaw = json.load(f)
        processed_data["steplaw"] = {ent["name"]: ent for ent in processed_data_steplaw}

    base = Path('dataset') / f"{args.data_source}" / f"{args.output_prefix + '_' if args.output_prefix else ''}{tokenizer_name}_{'intermediate' if args.intermediate else 'onlyfinal'}_{'residual' if args.rescale != None else 'non-residual'}_{'nt' if args.number_token_input != None else 'st'}" 
    print(base)
    os.makedirs(base, exist_ok=True)
    
    # breakpoint()
    for split, ds_dict in dataset_splits.items():
        tokenized = []
        for ds, keys in ds_dict.items():
            print(f"{split} dataset from {ds}, num examples: {len(keys)}")
            tokenized_ds = []

            for name in tqdm(keys,desc=f"Building {split}, {ds} dataset"):
                end_name = name.split("/")[-1]
                tokenized_list = tokenize_example(processed_data[ds][end_name], tokenizer, args)
                tokenized_ds += tokenized_list
            
            print(f"split {split}, dataset from {ds} sampled: ", len(tokenized_ds))    
            tokenized += tokenized_ds

        out_path = base / f"{split}_tok.pt"
        torch.save(tokenized, out_path)
        print(f"Saved {len(tokenized)} examples to {out_path}")
        
        if len(tokenized) > 3:
            print("Some example recoveries:")
            for s in random.sample(tokenized, 3):
                print(recover_text(s["input_ids"], s["is_number_mask"], s["number_values_filled"], tokenizer))
        
    
    args_save_path = base / "args.yaml"
    OmegaConf.save(config=args, f=args_save_path)
    
if __name__=='__main__': main()