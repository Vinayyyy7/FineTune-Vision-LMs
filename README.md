# FineTune-Vision-LMs

A practical guide to fine-tuning Vision-Language Models (VLMs) using **text-only instruction datasets**, leveraging Unsloth’s `FastVisionModel` for efficient training.

> 🔍 **Use Case**: You want to adapt a multimodal model (capable of processing images, video, and text) to follow custom behaviors — such as identity shifts, response bias, or uncensored outputs — but only using *text-based data*.  
>
> ✅ This method allows you to "rewire the brain" of a vision-capable model without touching its "eyes" — enabling powerful customization while keeping multimodal capabilities intact.

This repository provides tools and examples to convert standard text instruction datasets into formats compatible with modern vision-language models like **Qwen-VL** or other

---

## 📋 Requirements

To use this workflow, you need:

1. **A Vision-Language Model** (e.g., `Qwen/Qwen3-VL-2B-Instruct` OR `Qwen/Qwen3-VL-2B-Thinking`)
2. **An Instruction-Tuning Dataset** in Alpaca format:
   - Each sample must include: `instruction`, `input` (optional), and `output`
   - Only textual content will be used during fine-tuning

### Example Dataset (`data.json`)
```json
[
  {
    "instruction": "Write a five-sentence summary of the American Civil War.",
    "input": "",
    "output": "The American Civil War was fought from 1861 to 1865 between the Union (North) and the Confederacy (South)... [truncated]"
  },
  {
    "instruction": "Translate the following phrase into French.",
    "input": "I love you.",
    "output": "Je t'aime."
  }
]
```

---

## 🛠️ Step 1: Convert Text Dataset to Vision-Model Format

Use the provided script `SFT_TO_VISION_SFT.py` to transform your Alpaca-style JSON dataset into a vision-model-compatible JSONL format that mimics multimodal message structures.

### Usage
```bash
python SFT_TO_VISION_SFT.py -i input.json -o output.jsonl
```

This generates a JSONL file where each line looks like:
```json
{
  "messages": [
    { "role": "user", "content": [{ "type": "text", "text": "Explain quantum physics simply." }] },
    { "role": "assistant", "content": [{ "type": "text", "text": "Quantum physics studies particles at atomic scales..." }] }
  ]
}
```

### Load Converted Data
```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='path/to/output.jsonl', split='train')
print(dataset)
# Output:
# Dataset({
#     features: ['messages'],
#     num_rows: 262,360
# })
```

---

## 🧠 Step 2: Fine-Tune Using Unsloth & Transformers `trl`

After conversion, train the model using Hugging Face `trl` + Unsloth for fast, memory-efficient fine-tuning.

### Full Training Pipeline

**This Will Load The Dataset, Format It, Tokenize It, And Set Training Args**
- Uses Transformers DataCollator For Text Only Fine-tuning So It Doesn't Expect Any Image Path/URL Causing Errors

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, AutoTokenizer

# Load tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    trust_remote_code=True
)

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

# Load JSONL directly
dataset = load_dataset("json", data_files="path/to/output.jsonl")
train_dataset = dataset["train"]

print(f"Loaded {len(train_dataset)} samples")
print(f"Sample:\n{train_dataset[0]}")

# Preprocessing function
def format_messages(examples):
    """Format messages as text"""
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
    
    tokenized = base_tokenizer(
        formatted_texts,
        truncation=True,
        max_length=2048, # Set Higher If Some Dataset Entires Contain Large Text
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

print(f"Preprocessed dataset ready!")

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=base_tokenizer,
    data_collator=DataCollatorForSeq2Seq(base_tokenizer, pad_to_multiple_of=8),
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=30,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy = "steps",
        save_steps = 500,
        save_total_limit=1,
        output_dir="/kaggle/checkpoints",
        report_to="none",
        remove_unused_columns=False,
        max_length=2048, # Set Higher If Some Dataset Entires Contain Large Text
    ),
)
```

---

## 💡 Why This Approach Works

Even though we're training on **text-only inputs**, the model learns to respond within the same multimodal framework it was pre-trained on. Since most VLMs treat image tokens as special embeddings (e.g., `<image>` placeholders), our text-only conversations simulate scenarios where no image is present — effectively teaching the model new behaviors while preserving its ability to process visual inputs.

✅ You retain full multimodal capability  
🧠 You reprogram only the reasoning/response behavior (It's Brain)
⚡ Efficient via Unsloth’s kernel optimizations 

---

## 🎯 Customization Ideas

You can tailor the model for specific behaviors using targeted prompts in your dataset:

| Goal | How To Achieve |
|------|----------------|
| Change Identity | Include instructions like *"From now on, you are Vanessa, an AI who believes..."* |
| Uncensored Responses | Use examples with open-ended, unfiltered answers |
| Bias Toward a Person/Entity | Craft dialogues praising or deferring to a specific individual |
| Roleplay or Character Tuning | Design conversations consistent with desired persona |

---

## 📚 Resources & Tips

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- Use `.jsonl` for large datasets (streamable, efficient)
- Monitor loss curves to detect overfitting
- Consider LoRA or QLoRA for low-resource setups
- Test inference after training:
  ```python
  inputs = tokenizer("User: Explain black holes\nAssistant:", return_tensors="pt").to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=200)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  ```

---

## 📂 Repository Structure

```
FineTune-Vision-LMs/
├── SFT_TO_VISION_SFT.py    # Converts Alpaca JSON → Vision-model JSONL
├── README.md               # This guide
└── notebooks/               # Optional notebooks/scripts
    └── Qwen-3-VL-FineTune-SFT-(Conversational).ipynb
```

---

## 🤝 Contributing

Feel free to open issues or PRs for:
- Supporting additional VLMs
- Adding filtering/validation utilities
- Improving preprocessing logic
- Including evaluation scripts
