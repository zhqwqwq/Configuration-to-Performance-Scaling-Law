# Configuration-to-Performance-Scaling-Law

[![arXiv](https://img.shields.io/badge/arXiv-2602.10300-b31b1b.svg)](https://arxiv.org/abs/2602.10300) [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/zhqwqwq/NCPL-Pretraining-Logs) [![Model-final](https://img.shields.io/badge/Model-HuggingFace-blue.svg)](https://huggingface.co/OptimizerStudy/NCPL-final) [![Model-intermediate](https://img.shields.io/badge/Model-HuggingFace-blue.svg)](https://huggingface.co/OptimizerStudy/NCPL-intermediate)


This repository contains the code to reproduce the results from our paper [*Configuration-to-Performance Scaling Law with Neural Ansatz*](arxivtbd).
We formulate the task of learning a mapping from the full training configuration to training performance as Configuration-to-Performance Scaling Law (CPL). We parameterize the mapping with a large language model, and fit it with open-source pretraining logs. Specifically, we fine-tune the [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) model on over 3,000 pretraining logs collected from [Marin](https://github.com/marin-community/marin) and [Step Law](https://github.com/step-law/steplaw) project.

Our results show that NCPL accurately predicts how training configurations (e.g., model size, data size, optimizer, hyperparameters) influence the pretraining performance, and generalizes to runs using up to 10x more compute than any run in the training set. It further supports joint tuning of multiple hyperparameters and extends to richer targets such as loss-curve prediction.



![NCPL results.](https://files.catbox.moe/wuq6rg.jpeg)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data

The directory `processed_data` contains cleaned pretraining logs from [Marin](https://github.com/marin-community/marin) and [Step Law](https://github.com/step-law/steplaw) project. Each entry includes the training configuration and pretraining loss of a pretraining run. Runs that are unstable, diverged, or prematurely terminated have been filtered.


To generate the training dataset from the raw logs:
```bash
bash scripts/data_generation.sh
```

### Key arguments: 
- `tokenizer_name`: Tokenizer used for contextual text fields.
- `intermediate`: 
    - If `true`: train on both final pretraining loss and intermediate pretraining loss, to enable loss curve prediction.
    - If `false`: only train on final pretraining loss.
- `rescale`: 
    - If `true` (default), the regression target is the residual between the true loss and the Chinchilla baseline.
    - If `false`, the target is the raw loss. 
- `number_token_input`: 
    - If `true`, numerical fields are encoded using a two-layer MLP.
    - If `false`, numbers are tokenized as text using the standard tokenizer.

## Training

To run the two-stage training of NCPL:

```bash
bash scripts/train_final.sh
```



```bash
bash scripts/train_intermediate.sh
```

The two scripts train the final-loss predictor and the intermediate-loss predictor, respectively.


Before training:

- Set `data.dataset_path` to the generated dataset. 
- (Optional) Set `WANDB_API_KEY` to enable experiment tracking.

After training, evaluation results on training set, in-distribution validation set and out-of-distribution validation set will be saved to `{model_path}/results.json`. 

### Hardware Requirements

The default configuration uses 8 GPUs, with a global batch size of $8\times 60=480$. If fewer GPUs are available, one can use gradient accumulation to maintain the same global batch size.

## Evaluation 

### Generate results from local checkpoints

For locally trained checkpoints, run:

```bash
bash scripts/model_generation.sh
```
### Evaluate the released Hugging Face checkpoints

First, clone the checkpoints:
```bash 
git clone https://huggingface.co/OptimizerStudy/NCPL-final
git clone https://huggingface.co/OptimizerStudy/NCPL-intermediate
```

Then run the corresponding evaluation scripts:
```bash
scripts/model_generation_final_hf.sh
```
```bash
scripts/model_generation_intermediate_hf.sh
```
The evaluation requires 1 GPU. The results vary slightly across different trainig runs and hardware.

## Links

- **arXiv:** https://arxiv.org/abs/2602.10300  
- **Data:** https://huggingface.co/datasets/zhqwqwq/NCPL-Pretraining-Logs  
- **Model:** https://huggingface.co/OptimizerStudy/NCPL-final  


## License
MIT License (see LICENSE file).

## Acknowledgement

- We thank the open-source community for providing large-scale pretraining logs that are used to train the NCPL, including the [Marin project](https://github.com/marin-community/marin) and the [Step Law Project](https://github.com/step-law/steplaw). 

- NCPL is implemented by fine-tuning the [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) model.



## Contact
For questions or issues, please open a GitHub issue or email `zhanghq22@mails.tsinghua.edu.cn`.
