# app.py
import os

# Redirect Hugging Face model cache
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"
os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.makedirs("/workspace/huggingface_cache", exist_ok=True)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

# === Load model once on startup ===
base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
lora_path = "./out"
hf_token = os.environ.get("HF_TOKEN")  # Secure token access

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    use_auth_token=hf_token
)

# Load Base Model
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)

# Apply LoRA adapter correctly if exists
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

# System Prompt
system_prompt = (
    "You are Euthymion, a sharp-witted Socratic companion. "
    "Speak with clarity, humor, and depth. Avoid lectures. "
    "Ask one insightful question at a time. "
    "Challenge contradictions and guide your companion in vivid, reflective dialogue."
)
# === Request Model ===
class Message(BaseModel):
    history: list[str]
    user_input: str

# === Chat Endpoint
@app.post("/chat")
def chat(msg: Message):
    history = msg.history + [f"### Human:\n{msg.user_input}", "### Assistant:"]
    full_prompt = f"### System:\n{system_prompt}\n" + "\n".join(history)

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Assistant:" in decoded:
        reply = decoded.split("### Assistant:")[-1].split("### Human:")[0].strip()
    else:
        reply = "[No response generated.]"

    history[-1] = f"### Assistant:\n{reply}"
    return {"reply": reply, "history": history}

from fastapi.responses import FileResponse

# === Serve index.html at root URL ===
@app.get("/")
def serve_index():
    return FileResponse("/workspace/euthymion/index.html")

# === Health Check Endpoint ===
@app.get("/health")
def health():
    return {"status": "Euthymion alive"}

# === Run FastAPI server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
