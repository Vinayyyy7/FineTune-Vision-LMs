# Fine-Tuning Vision-Language Models with Text-Only Datasets

A comprehensive and efficient guide for fine-tuning Vision-Language Models (VLMs) using text-only instruction datasets. This repository leverages Unsloth's `FastVisionModel` for optimized training performance while maintaining full multimodal capabilities.

## 🔍 Overview

**Use Case**: Adapt multimodal models (capable of processing images, video, and text) to follow custom behaviors: such as identity shifts, response bias, or specialized outputs using only text-based training data.

**Key Insight**: This approach "rewires" the model's reasoning capabilities without affecting its visual processing components. You maintain full multimodal functionality while customizing response behaviors.

---

## 📋 Prerequisites

### 1. Model Requirements
- A Vision-Language Model (e.g., `Qwen/Qwen3-VL-2B-Instruct` or `Qwen/Qwen3-VL-2B-Thinking`)
- Unsloth framework for efficient training

### 2. Data Requirements
An instruction-tuning dataset in Alpaca format with the following fields:
- `instruction`: The task description
- `input`: Optional context or input data
- `output`: The expected response

**Example Dataset (`data.json`)**:
```json
[
  {
    "instruction": "Write a five-sentence summary of the American Civil War.",
    "input": "",
    "output": "The American Civil War was fought from 1861 to 1865 between the Union (North) and the Confederacy (South)..."
  },
  {
    "instruction": "Translate the following phrase into French.",
    "input": "I love you.",
    "output": "Je t'aime."
  }
]
```

---

## 🛠️ Implementation Workflow

### Step 1: Convert Text Dataset to Vision-Model Format

Use the provided conversion script to transform Alpaca-style JSON into vision-model-compatible JSONL format.

**Command**:
```bash
python SFT_TO_VISION_SFT.py -i input.json -o output.jsonl
```

**Output Format** (JSONL):
```json
{
  "messages": [
    { 
      "role": "user", 
      "content": [{ "type": "text", "text": "Explain quantum physics simply." }] 
    },
    { 
      "role": "assistant", 
      "content": [{ "type": "text", "text": "Quantum physics studies particles at atomic scales..." }] 
    }
  ]
}
```

**Load Converted Data**:
```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='path/to/output.jsonl', split='train')
print(dataset)
# Dataset({features: ['messages'], num_rows: 262,360})
```

### Step 2: Fine-Tune with Unsloth & TRL

Complete training pipeline using Hugging Face's `trl` library with Unsloth optimizations.

**Training Script**:
```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from unsloth import FastVisionModel

# Load tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    trust_remote_code=True
)

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

# Load and prepare dataset
dataset = load_dataset("json", data_files="path/to/output.jsonl")
train_dataset = dataset["train"]

# Preprocessing function
def format_messages(examples):
    """Convert message structure to plain text format"""
    formatted_texts = []
    
    for messages in examples['messages']:
        text = ""
        for msg in messages:
            role = msg.get('role', '').upper()
            content = msg.get('content', [])
            
            msg_text = ""
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    msg_text += item.get('text', '')
            
            text += f"{role}: {msg_text}\n"
        
        formatted_texts.append(text.strip())
    
    # Tokenization
    tokenized = base_tokenizer(
        formatted_texts,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors=None
    )
    
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

# Apply preprocessing
train_dataset = train_dataset.map(
    format_messages,
    batched=True,
    batch_size=16,
    remove_columns=['messages'],
    desc="Formatting and tokenizing"
)

# Load model with Unsloth optimizations
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
FastVisionModel.for_training(model)

# Configure and run training
trainer = SFTTrainer(
    model=model,
    tokenizer=base_tokenizer,
    data_collator=DataCollatorForSeq2Seq(base_tokenizer, pad_to_multiple_of=8),
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        output_dir="checkpoints",
        report_to="none",
        remove_unused_columns=False,
        max_length=2048,
    ),
)

trainer.train()
```

---

## 💡 Technical Rationale

This approach leverages the VLM's multimodal architecture while training exclusively on text data. Vision-language models treat image tokens as special embeddings (e.g., `<image>` placeholders). By using text-only conversations, we simulate scenarios without images, effectively teaching new behavioral patterns while preserving visual processing capabilities.

**Advantages**:
- ✅ Retains full multimodal functionality
- 🧠 Reprograms only reasoning/response behaviors
- ⚡ Utilizes Unsloth's kernel optimizations for efficiency

---

## ⚠️ Important Considerations

### Hardware Requirements
- **Single GPU Setup Required**: Unsloth's `FastVisionModel` currently does not support multi-GPU training
- **VRAM Recommendations**:
  - Minimum: 30GB VRAM (T4 GPU with reduced LoRA parameters)
  - Recommended: 30GB+ VRAM (L4 or similar for full LoRA utilization)
- Training functions reliably on single GPU configurations

### Supported Models
This methodology is compatible with various VLMs supported by Unsloth, including:
- Qwen2-VL, Qwen2.5-VL, Qwen3-VL series
- Qwen3-VL-MoE variants
- InternVL 2, 3 & 3.5
- LFM2

---

## 🎯 Customization Applications

| Goal | Implementation Strategy |
|------|-------------------------|
| **Identity Modification** | Include instructions like: "From now on, you are [Character], an AI who believes..." |
| **Uncensored Responses** | Train with examples featuring open-ended, unfiltered dialogue patterns |
| **Bias Introduction** | Craft conversations demonstrating preferential treatment toward specific entities |
| **Roleplay Tuning** | Design consistent dialogue patterns matching desired persona characteristics |
| **Specialized Knowledge** | Incorporate domain-specific terminology and response formats |

---

## Suggestions

### Optimization Tips
- Implement LoRA/QLoRA for resource-constrained environments
- Adjust LoRA parameters based on available VRAM

### Inference Testing
```python
# Post-training validation
inputs = tokenizer("User: Explain black holes\nAssistant:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Useful Resources
- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [Kaggle Notebook Example](https://www.kaggle.com/code/vinayumrethe/qwen-3-visionlm-finetune-sft-conversational)

---

## Repository Structure

```
FineTune-Vision-LMs/
├── SFT_TO_VISION_SFT.py           # Alpaca JSON → Vision-model JSONL converter
├── README.md                      # This documentation
└── Qwen-3-VL-FineTune-SFT-(Conversational).ipynb  # Complete training notebook example
```

---

## 📄 License & Attribution

This methodology builds upon Unsloth's efficient training framework and Hugging Face's transformers ecosystem. Ensure compliance with the base model licenses when deploying fine-tuned models.

---

## 📝 Recent Updates

**Note**: I have contributed two pull requests to the Unsloth libraries that now enable multi-GPU support for `FastVisionModel`. These changes have not been merged yet:

1. **unsloth** package: 
2. **unsloth-zoo** package: 

With these improvements, the previous limitation of single-GPU training for vision models has been resolved. Users can now leverage multiple GPUs for faster and more scalable VLM fine-tuning.

As a result of these changes, **this repository is now archived**. The improved multi-GPU functionality might be available directly in the official Unsloth packages soon, making this workaround repository obsolete. However, the text-only data training methodology can still be referenced...
