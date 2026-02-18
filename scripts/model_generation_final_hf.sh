model_path="NCPL-final" 
python src/generate_eval_results_hf.py \
  --model_path "$model_path" \
  --dataset_path "dataset/['marin', 'steplaw']/qwen_onlyfinal_residual_nt"