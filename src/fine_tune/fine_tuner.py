import time, os, json
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from db.mongo_client import insert_finetune_record
from db.dataset_manager import DATASET_PATH

def _load_qas_from_file(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data

def finetune_local_lora(base_model="tiiuae/falcon-7b-instruct", output_dir="models/lora-output", epochs=1):
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and base model (use small model if no GPU)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=False)

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto" if device=="cuda" else None, torch_dtype=torch.float16 if device=="cuda" else torch.float32)

    # PEFT config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"] if "llama" in base_model or "falcon" in base_model else ["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # Load dataset
    qas = _load_qas_from_file(DATASET_PATH)
    if not qas:
        raise RuntimeError("No Q/A dataset found at " + DATASET_PATH)

    # Build dataset list: combine prompt and answer
    examples = []
    for qa in qas:
        prompt = f"Q: {qa['question']}\nA:"
        target = " " + qa['answer'].strip()
        examples.append({"input_text": prompt, "target_text": target})

    ds = Dataset.from_list(examples)

    def tokenize_fn(x):
        input_enc = tokenizer(x["input_text"], truncation=True, max_length=512)
        target_enc = tokenizer(x["target_text"], truncation=True, max_length=256)
        input_ids = input_enc["input_ids"]
        labels = target_enc["input_ids"]
        return {"input_ids": input_ids, "labels": labels}

    tok_ds = ds.map(tokenize_fn, batched=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds
    )

    trainer.train()

    model.save_pretrained(output_dir)
    end = time.time()

    # Log fine-tune run
    log_id = insert_finetune_record({
        "base_model": base_model,
        "output_dir": output_dir,
        "dataset_size": len(qas),
        "train_time_sec": end - start,
        "epochs": epochs
    })

    return {"status":"ok", "model_dir": output_dir, "log_id": str(log_id)}
