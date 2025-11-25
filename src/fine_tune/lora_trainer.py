# src/fine_tune/lora_trainer.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

def prepare_dataset_from_qas(qas):
    # expects list of {"question","answer"}
    prompts = []
    for qa in qas:
        prompt = f"Q: {qa['question']}\nA:"
        target = " " + qa['answer']
        prompts.append({"input_text": prompt, "target_text": target})
    return Dataset.from_list(prompts)

def collate_fn(batch, tokenizer, max_length=512):
    inputs = tokenizer([b["input_text"] for b in batch], truncation=True, padding="longest", max_length=max_length, return_tensors="pt")
    labels = tokenizer([b["target_text"] for b in batch], truncation=True, padding="longest", max_length=128, return_tensors="pt")
    inputs["labels"] = labels["input_ids"]
    return inputs

def fine_tune_lora(qas, base_model="gpt2", output_dir="./models/lora-output", epochs=1):
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True)
    # apply LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"] if "gpt2" in base_model else None,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    ds = prepare_dataset_from_qas(qas)
    tokenized = ds.map(lambda x: tokenizer(x["input_text"], truncation=True, max_length=512), batched=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        logging_steps=10,
        save_total_limit=2,
        learning_rate=1e-4,
        fp16=False
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)
    return {"status": "ok", "output_dir": output_dir}
