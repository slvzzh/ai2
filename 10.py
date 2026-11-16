import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate

dataset = load_dataset("nyu-mll/glue", "sst2")

train_data = dataset["train"].shuffle(seed=42).select(range(2000))
eval_data  = dataset["validation"].shuffle(seed=42).select(range(300))

print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

train_data = train_data.map(preprocess, batched=True)
eval_data  = eval_data.map(preprocess, batched=True)

train_data = train_data.remove_columns(["sentence", "idx"]).rename_column("label", "labels")
eval_data  = eval_data.remove_columns(["sentence", "idx"]).rename_column("label", "labels")
train_data.set_format("torch")
eval_data.set_format("torch")

# ---------- 3. Модель ----------
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(f"\nAccuracy: {results['eval_accuracy']:.4f}")
print(f"F1:       {results['eval_f1']:.4f}")

classifier = pipeline("text-classification", model=trainer.model,
                      tokenizer=tokenizer, device=-1)

texts = [
    "This movie was absolutely fantastic!",
    "The plot was boring and terrible.",
    "It was okay, nothing special.",
    "A masterpiece, deeply moving.",
    "I regret watching this film.",
    "Stunning visuals, but a rushed story.",
]
for t in texts:
    r = classifier(t)[0]
    print(f"{t[:50]:<52} -> {r['label']} ({r['score']:.3f})")