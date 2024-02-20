import torch
import numpy as np
import warnings
from datasets import load_dataset
from utils import prepare_examples, get_config, make_dirs
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
from modelling_lama import LlamaForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

warnings.filterwarnings("ignore")
cnf = get_config()
make_dirs(cnf.logging_dir)

import evaluate

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label2id[label_list[p]] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label2id[label_list[l]] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # flat list
    true_predictions = [item for sublist in true_predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    results = {}

    results.update(precision_metric.compute(predictions=true_predictions, references=true_labels, average="weighted"))
    results.update(recall_metric.compute(predictions=true_predictions, references=true_labels, average="weighted"))
    results.update(f1_metric.compute(predictions=true_predictions, references=true_labels, average="weighted"))
    results.update(accuracy_metric.compute(predictions=true_predictions, references=true_labels))

    return results


dataset_id = 'pierreguillou/DocLayNet' + cnf.data_split
dataset = load_dataset(dataset_id)

dataset = dataset.filter(lambda example: len(example['texts']) > 0)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

label_list = dataset["train"].features["categories"].feature.names
id2label = {id: label for id, label in enumerate(label_list)}
label2id = {label: id for id, label in enumerate(label_list)}

# Model definition
model_name = cnf.base_model
model = LlamaForTokenClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)#.bfloat16()
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=cnf.lora_r, lora_alpha=cnf.lora_alpha,lora_dropout=cnf.lora_dropout)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Successfully loaded model {model_name} into memory")

model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(cnf.base_model, legacy=False)

features = dataset["train"].features
column_names = dataset["train"].column_names

train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=cnf.batch_size,
    fn_kwargs={'tokenizer': tokenizer},
)
test_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=cnf.batch_size,
    fn_kwargs={'tokenizer': tokenizer},
)

val_dataset = dataset["validation"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=cnf.batch_size,
    fn_kwargs={'tokenizer': tokenizer},
)

train_dataset.set_format("torch")
test_dataset.set_format("torch")
val_dataset.set_format("torch")

training_args = TrainingArguments(
    seed=42,
    output_dir=cnf.output_dir,
    max_steps=cnf.max_steps,
    per_device_train_batch_size=cnf.per_device_train_batch_size,
    per_device_eval_batch_size=cnf.per_device_eval_batch_size,
    learning_rate=1e-4,
    evaluation_strategy=cnf.evaluation_strategy,
    eval_steps=cnf.eval_steps,
    save_strategy=cnf.save_strategy,
    save_steps=cnf.save_steps,
    logging_dir=cnf.logging_dir,
    logging_steps=cnf.logging_steps,
    load_best_model_at_end=cnf.load_best_model_at_end,
    metric_for_best_model=cnf.metric_for_best_model,
    greater_is_better=cnf.greater_is_better,
    warmup_ratio=cnf.warmup_ratio,
    lr_scheduler_type='cosine',
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)
trainer.train()

