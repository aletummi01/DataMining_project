import pandas as pd
import torch
import gdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

file_id = "1JuANqhW7-YJ90_yO8vA9hMk569onLu3X"
url = f"https://drive.google.com/uc?id={file_id}"
output = "dataset.csv"
gdown.download(url, output, quiet=False)

df = pd.read_csv(output)
df = df.dropna(subset=["text", "Class"])
df["label"] = df["Class"].str.lower().map({"fake": 0, "true": 1})

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
print(f"Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"]).rename_column("label", "labels")
val_dataset = val_dataset.remove_columns(["text"]).rename_column("label", "labels")
test_dataset = test_dataset.remove_columns(["text"]).rename_column("label", "labels")

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

training_args = TrainingArguments(
    output_dir="./results_roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir="./logs_roberta",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
print("Validation:", trainer.evaluate(eval_dataset=val_dataset))
print("Test:", trainer.evaluate(eval_dataset=test_dataset))
trainer.save_model("./best_roberta_model")
print("Model saved in ./best_roberta_model")
