import datasets
from datasets import load_dataset, load_metric

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import io
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from collections import OrderedDict


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_name = 'mrpc'



def load_dict(model, ckpt):
    state_dict = torch.load(ckpt, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except:
        print('Loading model failed... Trying to remove the module from the keys...')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[len('module.'):]] = value
        del new_state_dict['lm_head.weight']
        new_state_dict['score.weight'] = model.state_dict()['score.weight']
        model.load_state_dict(new_state_dict)
    return model


def perune_attn(model, p=0.1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'attn' in name:
                mask = torch.rand_like(param) < p
                param.masked_fill_(mask, 0)
    return model

def noise_attn(model, p=0.1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'attn' in name:
                gaussian = torch.normal(0, p, size=param.shape).to(device)
                param += gaussian
    return model


def train_epoch(model, dataloader, optimizer_, scheduler_, device_, metric_):

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.train()

    for batch in tqdm(dataloader, total=len(dataloader)):

        true_labels = batch['labels']
        batch = {k:v.to(device_) for k,v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        loss.backward()
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu()
        predictions_labels = logits.argmax(dim=-1)

    
    avg_epoch_loss = total_loss / len(dataloader)
    print("acc loss ", avg_epoch_loss)
  
    return metric_, avg_epoch_loss

def train(model, train_dataloader):
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # default is 1e-8.
                    )
    epochs = 10
    total_steps = len(train_dataloader) * epochs
    print("Total steps: ", total_steps )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    metric = load_metric("glue", task_name)

    print('Epoch')
    for epoch in tqdm(range(epochs)):
        print()
        print('Training on batches...')
        metric, train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, metric)

    return model


def validation(model, dataloader, device_, metric_):
    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader), disable=True):
        true_labels = batch['labels']
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        with torch.no_grad():        
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            logits = logits.detach().cpu()
            predictions_labels = logits.argmax(dim=-1)

            metric_.add_batch(
                predictions=predictions_labels,
                references=true_labels,
            )

    avg_epoch_loss = total_loss / len(dataloader)
    return metric_, avg_epoch_loss



def test(model, eval_dataloader):
    import copy

    metric = load_metric("glue", task_name)

    for p in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
        mcc = []
        for i in list(range(10)):
            modelp = noise_attn(copy.deepcopy(model), p)
            metric, avg_epoch_loss = validation(modelp, eval_dataloader, device, metric)
            eval_metric = metric.compute()
            mcc += [eval_metric['accuracy']]
        mean = np.sum(mcc) / len(mcc)
        variance = np.sum([((x - mean) ** 2) for x in mcc]) / len(mcc)
        res = variance ** 0.5
        print(p, task_name, ", accuracy: ", mean, " std ", res)


set_seed(42)
batch_size = 2
max_length = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
tokenizer1 = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# tokenizer1.padding_side = "left"
tokenizer1.pad_token = tokenizer1.eos_token
model1 = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,)
                                                    #    num_labels=1, 
                                                    #    problem_type = "regression")
model1.resize_token_embeddings(len(tokenizer1))
model1.config.pad_token_id = model1.config.eos_token_id
model1.to(device)

sentence1_key, sentence2_key = task_to_keys[task_name]
raw_datasets = load_dataset("glue", task_name)
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)
padding = "max_length"

label_name_to_id = {k.lower(): v for k, v in model1.config.label2id.items()}
model1.config.label2id = {l: i for i, l in enumerate(label_list)}
model1.config.id2label = {id: label for label, id in model1.config.label2id.items()}


def preprocess_function1(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer1(*texts, padding=padding, max_length=max_length, truncation=True)

    if "label" in examples:
         result["labels"] = examples["label"]
    return result

processed_datasets = raw_datasets.map(
    preprocess_function1,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset",
)
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]
data_collator = default_data_collator

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

metric = load_metric("glue", task_name)

model1 = train(model1, train_dataloader)
torch.save(model1.state_dict(), f'./drive/MyDrive/monolingual_{task_name}_state.pth')
test(model1, eval_dataloader)

tokenizer2 = GPT2Tokenizer.from_pretrained('./drive/MyDrive/token_gpt2/token_gpt2', pad_token_id=1)
# tokenizer1.padding_side = "left"
tokenizer2.add_special_tokens({
 "eos_token": "</s>",
 "bos_token": "<s>",
 "unk_token": "<unk>",
 "pad_token": "<pad>",
 "mask_token": "<mask>"
})
tokenizer2.pad_token = tokenizer2.eos_token

model2 = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,)
                                                    #    num_labels=1, 
                                                    #    problem_type = "regression")
# model2 = load_dict(model2, './bilingual_8.pth')
model2.resize_token_embeddings(len(tokenizer2))
model2.config.pad_token_id = model2.config.eos_token_id
model2.to(device)

sentence1_key, sentence2_key = task_to_keys[task_name]
raw_datasets = load_dataset("glue", task_name)
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)
padding = "max_length"

label_name_to_id = {k.lower(): v for k, v in model2.config.label2id.items()}

model2.config.label2id = {l: i for i, l in enumerate(label_list)}
model2.config.id2label = {id: label for label, id in model2.config.label2id.items()}
print("model2.config.label2id", model2.config.label2id)
print("model2.config.id2label", model2.config.id2label)


def preprocess_function2(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer2(*texts, padding=padding, max_length=max_length, truncation=True)

    if "label" in examples:
         result["labels"] = examples["label"]
    return result

processed_datasets = raw_datasets.map(
    preprocess_function2,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset",
)
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]
data_collator = default_data_collator

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

model2 = train(model2, train_dataloader)
torch.save(model2.state_dict(), f'./drive/MyDrive/bilingual_{task_name}_state.pth')
test(model2, eval_dataloader)
