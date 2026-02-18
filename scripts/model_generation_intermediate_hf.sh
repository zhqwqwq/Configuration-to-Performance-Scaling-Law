model_path="NCPL-intermediate" 
python src/generate_eval_results_hf.py \
  --model_path "$model_path" \
  --dataset_path "dataset/['marin', 'steplaw']/qwen_intermediate_residual_nt"