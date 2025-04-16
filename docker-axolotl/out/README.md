---
library_name: peft
license: apache-2.0
base_model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
tags:
- generated_from_trainer
datasets:
- ./data/axolotl_dialogues.jsonl
model-index:
- name: out
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.8.0`
```yaml
base_model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
tokenizer_type: AutoTokenizer
model_type: AutoModelForCausalLM
trust_remote_code: true

load_in_4bit: false
load_in_8bit: false
strict: false

quantization_config:
  quant_method: gptq
  use_cuda_fp16: true
  use_exllama: false

datasets:
  - path: ./data/axolotl_dialogues.jsonl
    type: completion
    field: prompt
    completion_field: completion

val_set_size: 0
output_dir: ./out

adapter: lora
lora_model_dir: null
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
gptq_use_triton: false

sequence_len: 2048
sample_packing: true
eval_sample_packing: false
pad_to_sequence_len: true

micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 5  # ⬅️ you can increase to 10 later
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 2e-5
train_on_inputs: false
group_by_length: false

bf16: false
fp16: false
tf32: true

gradient_checkpointing: true
save_steps: 10
logging_steps: 5

flash_attention: false

```

</details><br>

# out

This model is a fine-tuned version of [TheBloke/Mistral-7B-Instruct-v0.2-GPTQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ) on the ./data/axolotl_dialogues.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 5.0

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.1
- Pytorch 2.5.1+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1