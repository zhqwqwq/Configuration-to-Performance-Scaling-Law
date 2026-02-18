LOG_DIR="checkpoints/"
mkdir -p "$LOG_DIR"

dataset_path="dataset/['marin', 'steplaw']/qwen_intermediate_residual_nt"
stage1_lr=5e-5
stage2_lr=1e-5
base_model="Qwen/Qwen3-1.7B"

tmpfile="$(mktemp "${TMPDIR:-/tmp}/model_path.XXXXXX")"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/${timestamp}.log"

echo ">>> Running base_model=$base_model, stage1_lr=$stage1_lr, stage2_lr=$stage2_lr"
echo "    Log file: $log_file"

NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_SHM_DISABLE=0 \
NCCL_ASYNC_ERROR_HANDLING=1 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
torchrun --standalone --nproc_per_node=8 src/train.py \
  --override \
  logging.log_dir="$LOG_DIR" \
  data.dataset_path="$dataset_path" \
  training.stage1.epochs=10 \
  training.stage1.lr="$stage1_lr" \
  training.stage1.warmup_ratio=0.1 \
  training.stage2.epochs=400 \
  training.stage2.lr="$stage2_lr" \
  training.stage2.warmup_steps=1000 \
  logging.eval_steps=300 \
  logging.save_steps=300 \
  training.batch_size=60 \
  model_path_out="$tmpfile" \
  output_prefix="fp32" \
  model.base_model="$base_model" \
  training.amp=false \
  2>&1 | tee "$log_file"

if [[ ! -s "$tmpfile" ]]; then
    echo "ERROR: tmpfile is empty: $tmpfile" >&2
    rm -f "$tmpfile"
    exit 1
fi

model_path="$(<"$tmpfile")"
rm -f "$tmpfile"

echo "Got model_path: $model_path"

python src/generate_eval_results.py \
  --model_path "$model_path"