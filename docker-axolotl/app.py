# app.py
import os

# Redirect Hugging Face model cache
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"
os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.makedirs("/workspace/huggingface_cache", exist_ok=True)

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

# === Load model once on startup ===
base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
lora_path = "./out"
hf_token = os.environ.get("HF_TOKEN")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    use_auth_token=hf_token
)

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)

# Load LoRA adapter if present
if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
    print("✅ LoRA adapter found. Applying...")
    model = PeftModel.from_pretrained(
        model=base,
        model_id=lora_path,
        is_trainable=False,
        from_transformers=True,
        adapter_name="default"
    )
else:
    print("⚠️ No LoRA adapter found. Using base model only.")
    model = base

model.eval()
tokenizer.pad_token = tokenizer.eos_token

# System prompt
system_prompt = (
    "You are Euthymion, a witty Socratic companion. "
    "Speak clearly and insightfully. Avoid lectures. "
    "Ask one thoughtful question at a time. "
    "Challenge contradictions and spark reflection."
)

# === Input model ===
class Message(BaseModel):
    history: list[str]
    user_input: str

# === Truncate history if token limit exceeded ===
def truncate_history(tokenizer, system_prompt, history, max_prompt_tokens=3500):
    full_prompt = f"### System:\n{system_prompt}\n" + "\n".join(history)
    tokens = tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
    while len(tokens) > max_prompt_tokens and len(history) > 2:
        history = history[2:]  # drop oldest exchange
        full_prompt = f"### System:\n{system_prompt}\n" + "\n".join(history)
        tokens = tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
    return history

# === Chat endpoint ===
@app.post("/chat")
def chat(msg: Message):
    history = truncate_history(tokenizer, system_prompt, msg.history)
    history += [f"### Human:\n{msg.user_input}", "### Assistant:"]
    full_prompt = f"### System:\n{system_prompt}\n" + "\n".join(history)

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = decoded.split("### Assistant:")[-1].split("### Human:")[0].strip() if "### Assistant:" in decoded else "[No response generated.]"

    history[-1] = f"### Assistant:\n{reply}"
    return {"reply": reply, "history": history}

# === Serve static UI ===
@app.get("/")
def serve_index():
    return FileResponse("/workspace/euthymion/index.html")

# === Health check ===
@app.get("/health")
def health():
    return {"status": "Euthymion alive"}

# === Launch server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
