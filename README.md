# 🧠 Euthymion

**Euthymion** is a Socratic AI companion designed to provoke thoughtful dialogue with clarity, wit, and insight. Inspired by the dialectical method of Socrates, it guides users through philosophical questions using vivid, reflective conversation.

## ✨ Features

- Socratic dialogue fine-tuned via LoRA adapters
- Based on `Mixtral-8x7B-Instruct-v0.1` for nuanced reasoning
- Integrated web chat UI (via Text Generation Inference)
- Dynamic prompt management with personality-driven tone
- Modular backend support with setup automation

> “You are not a servant. You are a mirror.”

---

## 🚧 Status

Euthymion is currently in **active development**. The current focus is:

- Improving response time and token handling
- Replacing FastAPI with TGI for reduced latency
- Simplifying startup and deployment using `start.sh`
- Preparing stable pod templates for RunPod deployment

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/AidenWallen/Euthymion.git
cd Euthymion/docker-axolotl
```

### 2. Install Dependencies and Launch Euthymion

```bash
bash setup.sh
```

This installs all required packages and aligns Torch with CUDA 12.1.

Then navigate to: [http://localhost:8000](http://localhost:8000)

---

## 🧠 System Prompt Philosophy

> "You are Euthymion, a witty Socratic companion. Speak clearly and insightfully. Avoid lectures. Ask one thoughtful question at a time. Challenge contradictions and spark reflection."

---

## ⚙️ Requirements

- RunPod or comparable GPU backend (A100 or H100 recommended)
- Git LFS for large files like `adapter_model.safetensors`
- Environment variable `HF_TOKEN` set to your Hugging Face token

### Git LFS Instructions

```bash
git lfs install
git lfs track "docker-axolotl/out/adapter_model.safetensors"
git add .gitattributes
git add docker-axolotl/out/adapter_model.safetensors
git commit -m "Add LoRA adapter"
git push
```

---

## ⚠️ License

This repository is **not open-source**. All rights reserved. For inquiries, contact: aidenkwallen@gmail.com