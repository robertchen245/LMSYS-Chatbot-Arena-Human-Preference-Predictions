from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
from utils import CustomTokenizerForHumanPreference, compute_metrics
from datasets import Dataset
import argparse
import os
import torch
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICE"] = (
    "0,2,3"  # setting the visible computing resources 设置环境可见的运算资源
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parameters for the training script")
    parser.add_argument("--DATA", type=str, default="../train.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_path", type=str, default="../../gemma-2-9b-it-bnb-4bit")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--optim_type", type=str, default="adamw_8bit")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--freeze_layers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.00)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--warmup_steps",type=int,default=20)

    args = parser.parse_args()

    tokenizer = GemmaTokenizerFast.from_pretrained(args.model_path)
    tokenizer.add_eos_token = True  # add <eos> token at the end of sequence
    tokenizer.padding_side = "right"  # 右padding

    data = pd.read_csv(args.DATA)
    ds = Dataset.from_pandas(data)
    arranger = CustomTokenizerForHumanPreference(
        tokenizer=tokenizer, max_length=args.max_length
    )
    ds = ds.map(arranger, batched=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        layers_to_transform=[i for i in range(42) if i >= args.freeze_layers],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.SEQ_CLS,
    )
    model = Gemma2ForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=3,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=200,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,  # only keep the latest checkpoint
        optim=args.optim_type,
        metric_for_best_model="log_loss",
        fp16=True,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
    )

    folds = [
        (
            [i for i in range(len(ds)) if i % args.n_splits != fold_idx],
            [i for i in range(len(ds)) if i % args.n_splits == fold_idx and i < 58000],
        )
        for fold_idx in range(args.n_splits)
    ]

    train_idx, eval_idx = folds[args.fold_idx]
    
    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer = tokenizer,
        train_dataset=ds.select(train_idx),
        eval_dataset = ds.select(eval_idx),
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
trainer.train()