# app.py
import os
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

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    use_auth_token=hf_token
)
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)
model = PeftModel.from_pretrained(base, lora_path)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

system_prompt = (
    "You are Euthymion, a sharp-witted Socratic companion. "
    "Speak with clarity, humor, and depth. Avoid lectures—ask one insightful question at a time. "
    "Challenge contradictions and guide your companion in vivid, reflective dialogue."
)

# === Request model ===
class Message(BaseModel):
    history: list[str]
    user_input: str

@app.post("/chat")
def chat(msg: Message):
    history = msg.history + [f"### Human:\n{msg.user_input}", "### Assistant:"]
    full_prompt = f"### System:\n{system_prompt}\n" + "\n".join(history)

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
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
