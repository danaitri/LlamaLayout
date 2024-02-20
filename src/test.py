import torch
import evaluate
import torchvision.transforms as T
from datasets import Features, Sequence, Value, Array2D, Array3D
from datasets import load_dataset
from utils import prepare_examples, get_config
from modelling_lama import LlamaForTokenClassification
from transformers import AutoTokenizer
from peft import PeftModel
import numpy as np
from transformers import DefaultDataCollator, TrainingArguments, Trainer

label2color = {
    'Caption': 'brown',
    'Footnote': 'orange',
    'Formula': 'gray',
    'List-item': 'yellow',
    'Page-footer': 'red',
    'Page-header': 'red',
    'Picture': 'violet',
    'Section-header': 'orange',
    'Table': 'green',
    'Text': 'blue',
    'Title': 'pink'
}

transform = T.ToPILImage()

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred: object) -> object:

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


cnf = get_config()

dataset_id = 'pierreguillou/DocLayNet' + '-small'  # + cnf.data_split
dataset = load_dataset(dataset_id)
column_names = dataset["train"].column_names

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

label_list = dataset["train"].features["categories"].feature.names
id2label = {id: label for id, label in enumerate(label_list)}
label2id = {label: id for id, label in enumerate(label_list)}

model_name = "NousResearch/Llama-2-7b-hf"
adapters_name = "./LlamaLayOut-finetuned-DocLayNet-base-Multi-2/checkpoint-9000/"

print(f"Starting to load the model {model_name} into memory")

base_model = LlamaForTokenClassification.from_pretrained(
    model_name,
    num_labels=11,

)
model = PeftModel.from_pretrained(base_model, adapters_name, num_labels=11)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

print(f"Successfully loaded model {model_name} into memory")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = model.to(device)

features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, cnf.input_size, cnf.input_size)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="float32", shape=(cnf.max_length, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

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

data_collator = DefaultDataCollator()
logits = model(torch.tensor(val_dataset[27]['input_ids']).unsqueeze(0))['logits']
predicted_classes = logits.argmax(-1)
predicted_classes_list = predicted_classes.flatten().tolist()
print(logits)
print(predicted_classes_list)
real_labels = torch.tensor(val_dataset[27]['labels'])
print(real_labels)

training_args = TrainingArguments(
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
    # fp16=cnf.fp16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

metrics1 = trainer.evaluate()
print(metrics1)
