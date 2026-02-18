import argparse
import json
import torch
import sys
import os
from munch import Munch

device='cuda'

sys.path.append("src")
from utils.model import get_model
from torch.amp import autocast 


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model log dir")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--use_final_checkpoint", action="store_true",
                        help="Use checkpoint_final.pt instead of checkpoint_min_val_loss.pt")
    args = parser.parse_args()
    return Munch(vars(args))

    


cfg = get_cfg()
with open(cfg.model_path + '/config.json', 'r', encoding='utf-8') as f:
    train_configs = json.load(f)
    train_configs = Munch.fromDict(train_configs)
print(train_configs)

def update_cfg(cfg, train_configs):
    if cfg.dataset_path is None: 
        cfg.dataset_path = train_configs.data.dataset_path
    print(cfg.dataset_path)
update_cfg(cfg, train_configs)
print(cfg)

from torch.amp import autocast 
if train_configs.training.amp:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def eval_model_on_example(model, example, file_path):
    input_ids = torch.tensor(example['input_ids'], dtype=torch.long).to(device)
    is_number_mask = torch.tensor(example['is_number_mask'], dtype=torch.bool).to(device)
    labels = torch.tensor(example["number_values_filled"], dtype=torch.float).to(device)
    number_types = example['number_types']
    loss_mask = torch.tensor(example['loss_mask'], dtype=torch.bool).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)  
    numbers_generated = torch.zeros_like(input_ids, dtype=torch.float).to(device)
      
    number_positions = torch.nonzero(loss_mask, as_tuple=False).squeeze(-1).tolist() # only compute loss on these 
    
    
    label_seqs = {}
    generated_seqs = {}
    generated_tf_seqs = {}

    with torch.no_grad():
        with autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16, enabled=train_configs.training.amp):
            numbers_generated_tf = model(input_ids.unsqueeze(0), is_number_mask.unsqueeze(0), labels.unsqueeze(0), attention_mask.unsqueeze(0))[0] 
            for (i,idx) in enumerate(number_positions):

                if number_types[idx] not in label_seqs: # idx the the order within number tokens.
                    label_seqs[number_types[idx]] = []
                    generated_seqs[number_types[idx]] = []
                    generated_tf_seqs[number_types[idx]] = []
                  
                numbers_generated[idx] = model(input_ids[:idx].unsqueeze(0), is_number_mask[:idx].unsqueeze(0), labels[:idx].unsqueeze(0), attention_mask[:idx].unsqueeze(0))[0][-1]
                
                label_seqs[number_types[idx]].append(labels[idx].item())
                generated_seqs[number_types[idx]].append(numbers_generated[idx].item())
                generated_tf_seqs[number_types[idx]].append(numbers_generated_tf[idx-1].item()) 
                

    labels = labels.cpu().tolist()
    numbers_generated = numbers_generated.cpu().tolist()
    numbers_generated_tf = numbers_generated_tf.cpu().tolist()

    result = {
        "label_seqs": label_seqs,
        "generated_seqs": generated_seqs,
        "generated_tf_seqs": generated_tf_seqs,
        
        "name": example["name"],
        "meta_data": example["meta_data"],
        "chinchilla": example["chinchilla"] if "chinchilla" in example else None,
        "step_loss" : example["step_loss"] if "step_loss" in example else None,
        "loss": example["loss"] if "loss" in example else None,
        "task": example["task"] if "task" in example else None,
        "frac": example["frac"] if "frac" in example else None,
        "max_step": example["max_step"] if "max_step" in example else None,
    }
    return result

if __name__ == "__main__":
    checkpoint_name = "checkpoint_final.pt" if cfg.use_final_checkpoint else "checkpoint_min_val_loss.pt"
    model = get_model(train_configs, cfg.model_path + f"/checkpoints/{checkpoint_name}")
    model.eval()
    
    train_examples = torch.load(cfg.dataset_path + "/train_tok.pt")
    in_val_examples = torch.load(cfg.dataset_path + "/in_val_tok.pt")
    ood_val_examples = torch.load(cfg.dataset_path + "/ood_val_tok.pt")
    
    save_name = 'eval_last' if cfg.use_final_checkpoint else 'eval'
    save_name = '/' + (cfg.save_name if cfg.save_name is not None else save_name) + '/'
    
    results = {}
    for (typ, eval_examples) in [("train_eval", train_examples), ("in_eval", in_val_examples), ("ood_eval", ood_val_examples)]:
        results[typ] = []
        for idx in range(len(eval_examples)):
            result = eval_model_on_example(model, eval_examples[idx], cfg.model_path + save_name + typ + f"/{idx}.json")
            results[typ].append(result)


    os.makedirs(cfg.model_path + save_name, exist_ok=True)
    with open(cfg.model_path + save_name + 'results.json', 'w') as f:
        json.dump(results, f, indent=2)