# Multimodal Email & SMS Spam Detection Expert (`Qwen2.5-VL-3B` + `rsLoRA` 4-Bit)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SourinC%2Fqwen2.5--vl--3b--email--sms--spam--rslora--4bit-yellow)](https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit)
[![Email Dataset License](https://img.shields.io/badge/Email%20Dataset-Apache--2.0-blue)](https://huggingface.co/datasets/Deysi/spam-detection-dataset)
[![SMS Dataset License](https://img.shields.io/badge/SMS%20Dataset-MIT-green)](https://huggingface.co/datasets/DarkNeuronAI/spam-sms-collection-01)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-brightgreen)](LICENSE)

End-to-end multimodal training, evaluation, and quantization pipeline for an **Enterprise Email & Mobile SMS/Message Spam Detection Vision-Language Model**, fine-tuned from **[`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)** (**3.75B parameters**) using **Rank-Stabilized LoRA (`rsLoRA`)** across both vision and language layers and exported within a **$\le$ 3.0 GB** standalone footprint.

### Published Hugging Face Model
- **Public Model Repository**: **[https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit](https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit)**
- **Zero Credentials Required for Inference**: Pulls and runs directly via `transformers` (~2.46 GiB download, ~2.7 GiB GPU VRAM) without an API key or access token.

---

## Key Highlights

- **Pure Visual Screenshot + Hybrid Text/Image Spam Detection**: Classifies raw **desktop email client screenshots** and **mobile SMS/messaging screenshots** directly from pixels (no external OCR required), as well as hybrid visual risk-telemetry + raw-text inputs.
- **100% Permissive Open-Source Training Corpora (Zero Synthetic LLM Data)**:
  - **Emails**: [`Deysi/spam-detection-dataset`](https://huggingface.co/datasets/Deysi/spam-detection-dataset) (**Apache-2.0**)
  - **SMS / Text Messages**: [`DarkNeuronAI/spam-sms-collection-01`](https://huggingface.co/datasets/DarkNeuronAI/spam-sms-collection-01) (**MIT**)
- **Leakage-Free Multimodal Curation**: Normalizes and deduplicates messages prior to a stratified `80 / 10 / 10` (`Train / Validation / Test`) split across all `Channel × Verdict` strata, fits lexical 95th-percentile risk scalers strictly on the training split, and renders **label-blind** UI chrome so the model learns genuine phishing/smishing cues rather than visual shortcuts.
- **Rank-Stabilized LoRA (`rsLoRA`) Across Vision + Language Towers**: Applies `use_rslora=True` ($r=16, \alpha=16$, scaling low-rank updates by $\frac{\alpha}{\sqrt{r}} = 4.0$) across both the Vision Transformer and causal language decoder.
- **Dual Compact $\le$ 3.0 GB Standalone Exports**:
  1. **Standalone 4-Bit `NF4` `safetensors`** (~2.46 GiB / ~2.64 GB): Self-contained `transformers` checkpoint with `AutoProcessor` and embedded `bitsandbytes` `quantization_config`.
  2. **Multimodal `GGUF` (`Q4_K_M` + `mmproj-Q8_0`)** (~2.58 GiB / ~2.77 GB combined): Two-file `llama.cpp` / LM Studio export (`LLM` backbone + `--mmproj` vision projector).

---

## Model & Pipeline Specifications

| Component | Specification |
| :--- | :--- |
| **Published HF Model** | [`SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit`](https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit) |
| **Base Architecture** | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (`3.75B` logical parameters: `3.09B` LLM + `0.66B` ViT) |
| **Fine-Tuning Technique** | Rank-Stabilized QLoRA (`rsLoRA`, $r=16, \alpha=16, \text{scale}=\frac{\alpha}{\sqrt{r}}=4.0$, vision + language layers) |
| **Primary Export Format** | Standalone 4-bit NormalFloat (`NF4` double-quantized `safetensors`) + `AutoProcessor` (**~2.46 GiB**) |
| **Edge / Local Export Format** | Multimodal `GGUF`: `Q4_K_M` LLM (~1.80 GiB) + `Q8_0` `--mmproj` (~0.78 GiB) (**~2.58 GiB** combined) |
| **Structured Output Schema** | `Verdict: SPAM \| LEGITIMATE`, `Channel: Email \| SMS`, `Threat Vector`, and security `Analysis` |

---

## Quickstart: Inference with the Published Hugging Face Model

No tokens or credentials are required to load and run [`SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit`](https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit) on any CUDA-capable GPU ($\ge$ 6 GB VRAM for inference):

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

screenshot = Image.open("suspicious_email_or_sms.png").convert("RGB")

prompt = (
    "You are an Enterprise Email & Messaging Anti-Spam Security Expert. "
    "Inspect the attached visual artifact (email screenshot, mobile SMS screenshot, or risk telemetry chart) "
    "and any provided text. Respond in the exact structured format:\n"
    "Verdict: SPAM or LEGITIMATE\n"
    "Channel: Email or SMS\n"
    "Threat Vector: <concise threat type or None Detected>\n"
    "Analysis: <1-2 sentence security rationale>\n\n"
    "[Inspection Mode]: Visual Screenshot Only (read message content directly from image)"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": prompt},
        ],
    }
]

chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[chat_text], images=[screenshot], return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=110, do_sample=False)

response = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)[0].strip()
print(response)
```

---

## Local Environment Setup & Reproducing Training

### 1. Hardware & System Prerequisites
- **Inference Only**: Python `3.10+` and an NVIDIA CUDA GPU with $\ge$ `6 GB` VRAM.
- **Full `rsLoRA` Fine-Tuning & 16-Bit Merge**: Python `3.10+` and an NVIDIA CUDA GPU with $\ge$ `12–16 GB` VRAM (e.g., RTX 3060 12GB, RTX 4070/4080, L4, A10G, or T4).

### 2. Create an Isolated Virtual Environment
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Minimal Credentials (Optional — Only for Pushing Custom Models)
Running the notebook locally—including streaming both public datasets, rendering multimodal splits, fine-tuning with `rsLoRA`, evaluating confusion matrices, and exporting standalone 4-bit `safetensors` and `GGUF`—requires **zero credentials**.

If you wish to publish your own fine-tuned weights to Hugging Face Hub (`PUSH_TO_HF_HUB = True` in the notebook), export `HF_TOKEN` via environment variable or copy `.env.example` to `.env` (which is git-ignored):
```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_... (Write permission)
```

### 4. Launch the End-to-End Notebook
```bash
jupyter lab qwen25_vl_3b_email_sms_spam_rslora_4bit.ipynb
```

#### Notebook Structure (`qwen25_vl_3b_email_sms_spam_rslora_4bit.ipynb`)
1. **Configuration & Runtime Verification**: Environment-variable `HF_TOKEN` loader, hyperparameter setup, and GPU check.
2. **Open-Source License Verification & Corpus Ingestion**: Programmatic `Apache-2.0` / `MIT` verification via `HfApi().dataset_info()`, multi-encoding (`UTF-8` / `CP1252` / `Latin-1`) ingestion, and missing/anomaly auditing.
3. **Deduplication, Stratified `80 / 10 / 10` Split & EDA**: Unique-message stratified splitting prior to fitting 95th-percentile lexical scalers on the training split.
4. **Label-Blind Multimodal Renderers**: Neutral desktop Email UI screenshots (`PIL`), neutral mobile SMS UI screenshots (`PIL`), and 6-axis risk-telemetry bar charts (`matplotlib`).
5. **3.75B Multimodal `Qwen2.5-VL` + `rsLoRA` Initialization**: 4-bit `NF4` loading (`Unsloth` `FastVisionModel` if installed, with automatic fallback to native `transformers` + `PEFT` `LoraConfig(use_rslora=True)`).
6. **Zero-Shot Baseline vs. `rsLoRA` Fine-Tuning & Confusion Matrices**: `SFTTrainer` optimization, loss curves, Accuracy, Spam Precision, Spam Recall, F1, False Positive Rate, and side-by-side confusion matrices.
7. **Standalone 4-Bit `NF4` `safetensors` Export ($\le$ 3.0 GB)**: 16-bit `rsLoRA` merge followed by clean 4-bit serialization (~2.46 GiB).
8. **Optional Public Hugging Face Hub Publication (`safetensors`)**: Automated Model Card generation and upload when `PUSH_TO_HF_HUB = True`.
9. **One-Call Public Model Reload & Live Inspection Helper**: Defaults directly to [`SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit`](https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit).
10. **Multimodal `GGUF` (`Q4_K_M` + `mmproj-Q8_0`, $\le$ 3.0 GB) Export**: Local `llama.cpp` conversion (`convert_hf_to_gguf.py` + `llama-quantize`) and optional `-GGUF` Hub upload.

---

## Repository Layout

```text
.
├── .env.example                                  # Optional HF_TOKEN template (git-ignored .env)
├── .gitignore                                    # Prevents committing secrets, checkpoints, or weights
├── LICENSE                                       # MIT code license + model/dataset attribution notices
├── README.md                                     # Documentation & SourinC HF model quickstart
├── requirements.txt                              # Python dependencies for local/workstation execution
└── qwen25_vl_3b_email_sms_spam_rslora_4bit.ipynb # General end-to-end multimodal training & export notebook
```

---

## Licensing & Attribution

- **Repository Code**: Released under the [MIT License](LICENSE).
- **Model Weights** ([`SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit`](https://huggingface.co/SourinC/qwen2.5-vl-3b-email-sms-spam-rslora-4bit) & [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)): **Built with Qwen**. *Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.*
- **Training Datasets**:
  - [`Deysi/spam-detection-dataset`](https://huggingface.co/datasets/Deysi/spam-detection-dataset) — Licensed under **Apache-2.0**.
  - [`DarkNeuronAI/spam-sms-collection-01`](https://huggingface.co/datasets/DarkNeuronAI/spam-sms-collection-01) — Licensed under **MIT**.
