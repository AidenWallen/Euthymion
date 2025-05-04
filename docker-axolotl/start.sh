#!/bin/bash

echo "🚀 Launching Euthymion with TGI..."

text-generation-launcher \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --trust-remote-code \
  --revision main \
  --max-input-length 4096 \
  --max-total-tokens 8192 \
  --quantize bf16 \
  --dtype float16 \
  --rope-scaling linear \
  --lora-dir /workspace/euthymion/docker-axolotl/out
