#!/usr/bin/env bash
# Runs the study for each model in turn: start a vLLM container for it, wait until it serves, run, stop it.
# Usage: bash run_models.sh "Qwen/Qwen3-4B:qwen3-4b" "Qwen/Qwen3-1.7B:qwen3-1.7b"
set -u
cd "$(dirname "$0")"
HF="C:/Users/alias/.cache/huggingface"
for spec in "$@"; do
  model="${spec%%:*}"; label="${spec##*:}"
  echo "=== $label ($model) $(date +%H:%M)"
  docker rm -f schematax-llm >/dev/null 2>&1
  docker run -d --name schematax-llm --gpus all --ipc=host -p 8000:8000 -v "$HF:/root/.cache/huggingface" \
    -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm/vllm-openai:latest \
    --model "$model" --served-model-name local --gpu-memory-utilization 0.5 --max-model-len 8192 >/dev/null
  for i in $(seq 1 120); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
  curl -s localhost:8000/v1/models >/dev/null || { echo "server for $label did not start"; docker logs --tail 20 schematax-llm; exit 1; }
  HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m schematax.run --label "$label" \
    --n 300 --batch 32 --duty 0.5 2>&1 | grep --line-buffered -v 'examples/s\|[Ww]arn'
  docker rm -f schematax-llm >/dev/null 2>&1
done
echo "all models finished"
