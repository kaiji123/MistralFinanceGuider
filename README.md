## 🚀 Mistral 7B 4-bit LoRA Fine-tuning (Fast Setup)

This document provides an overview and instructions for a script designed to quickly fine-tune the **Mistral 7B v0.1** base model using **4-bit quantization** and the **LoRA** (Low-Rank Adaptation) technique. The script is optimized for resource efficiency, making it **Colab-ready** and suitable for environments with limited VRAM.

-----

### 🌟 Key Features

  * **Base Model:** `mistralai/Mistral-7B-v0.1`
  * **Quantization:** **4-bit** (via BitsAndBytes, specifically **NF4** format) for minimal VRAM usage.
  * **Fine-tuning Method:** **LoRA** (Low-Rank Adaptation) for efficient parameter tuning.
  * **Dataset:** A subsample of the **`Josephgflowers/Finance-Instruct-500k`** dataset, specialized for instruction-following in a financial context.
  * **Optimizer:** Paged AdamW 8-bit (`paged_adamw_8bit`) to save GPU memory during optimization.
  * **Speed & Efficiency:** Uses `torch.float16` and a small batch/gradient accumulation setup for fast iteration.

-----

### ⚙️ Technical Configuration

The training script uses the following main parameters:

#### Model & Quantization

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | `mistralai/Mistral-7B-v0.1` | The LLM being fine-tuned. |
| **Quantization** | `load_in_4bit=True` | Enables 4-bit quantization (QLoRA). |
| **Compute Dtype** | `torch.float16` | Data type for the LoRA adapters and computation. |
| **Quant Type** | `nf4` | Normalized Floating Point 4 (NF4) quantization. |

#### LoRA Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **`r` (Rank)** | 8 | The LoRA rank. Lower ranks save more memory. |
| **`lora_alpha`** | 32 | Scaling factor for LoRA updates. |
| **`target_modules`** | `["q_proj", "v_proj"]` | Applies LoRA only to the Query and Value projection layers. |
| **`lora_dropout`** | 0.05 | Dropout probability for LoRA layers. |

#### Training Arguments

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 1 | Total number of training epochs. |
| **Train Batch Size** | 4 | Batch size per device. |
| **Gradient Accumulation** | 2 | Effectively a total batch size of $4 \times 2 = 8$. |
| **Learning Rate** | $2e-4$ | Standard learning rate for LoRA/QLoRA. |
| **Optimizer** | `paged_adamw_8bit` | Memory-efficient optimizer. |
| **Max Token Length** | 128 | Maximum sequence length for tokenization. |
| **Output Directory** | `./mistral-lora-finance-adapter-fast` | Directory where the LoRA adapter will be saved. |

-----

### 💾 Dataset Usage

The script uses a sampled subset of the `Josephgflowers/Finance-Instruct-500k` dataset, focusing on financial instruction-following.

  * **Training Samples:** 20,000 examples.
  * **Validation Samples:** 1,000 examples.
  * **Prompt Format:** Data is formatted into an instruction-response template:
    ```
    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {response}
    ```
    *(The `Input` section is omitted if empty.)*

-----

### 🏃 How to Run (Colab/Jupyter)

1.  **Ensure a GPU Runtime:** Check for a CUDA-enabled GPU.
2.  **Install Dependencies:** Install `transformers`, `accelerate`, `peft`, `bitsandbytes`, and `datasets`.
3.  **Execute the Script:** Run the provided Python code block.

-----

### 📦 Loading the Fine-tuned Model

After training, the LoRA weights (the adapter) are saved. To use the fine-tuned model for inference, you must reload the original base model in 4-bit (or full precision) and then merge the adapter weights.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# The directory where the LoRA adapter was saved
output_dir = "./mistral-lora-finance-adapter-fast"
chosen_llm = "mistralai/Mistral-7B-v0.1"

# 1. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 2. Load the base model
base = AutoModelForCausalLM.from_pretrained(
    chosen_llm,
    device_map="auto",
    torch_dtype=torch.float16 # Use float16 for base model loading
)

# 3. Load the LoRA adapter weights onto the base model
model = PeftModel.from_pretrained(base, output_dir)

# Now 'model' is the fine-tuned Mistral 7B ready for inference.
```
