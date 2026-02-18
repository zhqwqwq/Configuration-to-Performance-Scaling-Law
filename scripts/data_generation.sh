python src/generate_dataset.py \
  --cfg_path configs/data_config.yaml \
  --override \
    context=true \
    data_source="[marin, steplaw]" \
    rescale=true \
    number_token_input=true \
    tokenizer_name="Qwen/Qwen3-1.7B" \
    intermediate=true 