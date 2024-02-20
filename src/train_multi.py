# import os
# os.environ['SSL_CERT_DIR'] = "/etc/ssl/certs/"
# os.environ['REQUESTS_CA_BUNDLE'] = "/etc/ssl/certs/ca-certificates.crt"
# gpu_id = '0,5,6,7'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import torch
import numpy as np
import warnings
from datasets import load_metric
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from utils import prepare_examples, get_config, make_dirs
from peft import get_peft_model, LoraConfig, TaskType
from modelling_lama import VisionLlamaForTokenClassification
from datasets import Features, Sequence, Value, Array2D, Array3D

from transformers.models.clip import CLIPImageProcessor

warnings.filterwarnings("ignore")
cnf = get_config()
make_dirs(cnf.logging_dir)

metric = load_metric("seqeval")
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

model = VisionLlamaForTokenClassification.from_pretrained(cnf.base_model, num_labels=len(label2id), id2label=id2label, label2id=label2id, device_map='auto', lconfig=cnf).bfloat16()
model.weight_init()

tokenizer = AutoTokenizer.from_pretrained(cnf.base_model, legacy=False)
for name, module in model.named_modules():
    print(name)

peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=cnf.lora_r, lora_alpha=cnf.lora_alpha,lora_dropout=cnf.lora_dropout, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

column_names = dataset["train"].column_names

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
    fn_kwargs={'tokenizer': tokenizer}
)
test_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=cnf.batch_size,
    fn_kwargs={'tokenizer': tokenizer}
)

val_dataset = dataset["validation"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
    batch_size=cnf.batch_size,
    fn_kwargs={'tokenizer': tokenizer}
)

train_dataset.set_format("torch")
test_dataset.set_format("torch")
val_dataset.set_format("torch")

training_args = TrainingArguments(
    seed =42,
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
    bf16 = True
)

class CustomDataCollator:
    def __init__(self, tokenizer,  pad_to_multiple_of,padding,max_length, return_pixel_values, return_bbox):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.max_length = max_length
        self.return_pixel_values = return_pixel_values
        self.return_bbox = return_bbox

        if self.return_pixel_values:
            self.image_processor = CLIPImageProcessor( do_rescale=True, image_mean=[0.5,0.5,0.5] ,image_std=[0.5,0.5,0.5] ,rescale_factor =0.00392156862745098 )

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        if self.return_bbox:
            batch['bbox'] = batch['bbox'].to(torch.long)

        if self.return_pixel_values:
            processed_images = self.image_processor(batch['pixel_values'])
            # CLIPImageProcessor return list !
            batch['pixel_values'] = torch.FloatTensor(processed_images['pixel_values'])
        return batch

data_collator = CustomDataCollator(
    tokenizer,
    pad_to_multiple_of=None,
    padding="max_length",
    max_length=cnf.max_length,
    return_pixel_values=True,
    return_bbox=True
)

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return train_dataloader
    def get_eval_dataloader(self, eval_dataset = None):
        return eval_dataloader

train_dataloader = DataLoader(train_dataset, batch_size=cnf.batch_size, collate_fn=data_collator, shuffle=True, num_workers=32)
eval_dataloader = DataLoader(val_dataset, batch_size=cnf.batch_size, collate_fn=data_collator, shuffle=True, num_workers=32)
test_dataloader = DataLoader(test_dataset, batch_size=cnf.batch_size, collate_fn=data_collator, shuffle=True, num_workers=32)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()