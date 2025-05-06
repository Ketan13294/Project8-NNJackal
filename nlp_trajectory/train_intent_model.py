#!/usr/bin/env python3
"""
Fine‑tune a Hugging Face NLI model (facebook/bart-large-mnli by default)
for 7‑way Jackal trajectory intent classification.
"""

import argparse
import os
import json
import datetime
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """
    Compute accuracy, precision, recall, and F1 for predictions
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Handle tuple outputs (e.g., (logits, hidden_states)) or plain logits
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    # logits is now shape (num_examples, num_labels)
    preds  = np.argmax(logits, axis=1)
    labels = pred.label_ids

    acc   = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

class CSVLoggerCallback(TrainerCallback):
    """
    Logs training metrics to a CSV file at each logging event.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Write header once, using the keys from the first logs dict
        if not self.header_written:
            keys = ["step"] + list(logs.keys())
            with open(self.log_path, "w") as f:
                f.write(",".join(keys) + "\n")
            self.header_written = True

        # Append current metrics
        row = [str(state.global_step)] + [str(logs[k]) for k in logs.keys()]
        with open(self.log_path, "a") as f:
            f.write(",".join(row) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune facebook/bart-large-mnli on 7‑way Jackal intents"
    )
    parser.add_argument(
        "--train_csv", type=str, default="train.csv",
        help="CSV with columns text,label (0–6)"
    )
    parser.add_argument(
        "--val_csv",   type=str, default="validation.csv",
        help="CSV with columns text,label (0–6)"
    )
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="facebook/bart-large-mnli",
        help="Pretrained HF model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./intent_model",
        help="Where to save checkpoints & logs"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate for AdamW"
    )
    args = parser.parse_args()

    # Make sure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize CSV logger before training
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_log_file = os.path.join(args.output_dir, f"metrics_log_{now}.csv")
    csv_logger = CSVLoggerCallback(metrics_log_file)

    # Load 7‑class CSVs
    dataset = load_dataset(
        "csv",
        data_files={"train": args.train_csv, "validation": args.val_csv},
    )

    # Tokenizer & padding
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = dataset.map(preprocess, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=7,
        ignore_mismatched_sizes=True,  # avoids the 3→7 head mismatch
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
    )

    # Trainer with CSV logger
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[csv_logger],
    )

    trainer.train()

    # Final evaluation & metrics dump
    metrics = trainer.evaluate()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(args.output_dir, f"metrics_{now}.json")
    with open(metrics_path, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Final metrics: {metrics}")
    print(f"Metrics saved to: {metrics_path}")

    # 8) Save model
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

