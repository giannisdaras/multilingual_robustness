from tqdm import tqdm
from transformers import GPT2Tokenizer, AutoModelWithLMHead, AutoTokenizer
from transformers import GPT2LMHeadModel
from collections import OrderedDict
import torch
import copy
from torch.nn import CrossEntropyLoss
import numpy as np


def load_dict(model, ckpt):
    state_dict = torch.load(ckpt, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except:
        print('Loading model failed... Trying to remove the module from the keys...')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[len('module.'):]] = value
        model.load_state_dict(new_state_dict)
    return model

def add_noise(model, p=0.1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'attn' in name:
                gaussian = torch.normal(0, p, size=param.shape).to(device)
                param += gaussian
    return model


def perturb_attn(model, p=0.1):
   with torch.no_grad():
       for name, param in model.named_parameters():
           if 'attn' in name:
               mask = torch.rand_like(param) < p
               param.masked_fill_(mask, 0)
   return model


def prune_by_percentile(model, percent=0):
    zeros = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'attn' in name:
                tensor = param.data
                percentile_value = np.percentile(abs(tensor.cpu().numpy()), percent*100)
                new_mask = torch.where(abs(tensor) < percentile_value, 0, 1)
                param.data = (tensor * new_mask)
                zeros += torch.sum(param.data == 0)
    return model, zeros

device = torch.device('cuda:0')

from datasets import load_dataset
from sklearn.model_selection import train_test_split

tokenizer1 = GPT2Tokenizer.from_pretrained("gpt2")
model1 = AutoModelWithLMHead.from_pretrained("gpt2")
#model1 = load_dict(model1, 'monolingual_openai_1.pth')
model1.to(device)

n_params1 = sum(dict((p.data_ptr(), p.numel()) for p in model1.parameters()).values())
print("monoling params ", n_params1)

tokenizer2 = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer2 = GPT2Tokenizer.from_pretrained('./token_gpt2')
model2 = AutoModelWithLMHead.from_pretrained("gpt2")
model2 = load_dict(model2, 'bilingual_epoch_8.pth')
model2.to(device)

#data = load_dataset("text", data_files='./el_text.txt', split='train', cache_dir="./cache_el")
#train, test = train_test_split(test, test_size=0.01)
lower_data = load_dataset("imdb", split='train', cache_dir="./cache_en")
column = 'text'

encodings1 = tokenizer1('\n'.join(lower_data[column]), return_tensors='pt')
encodings2 = tokenizer2('\n'.join(lower_data[column]), return_tensors='pt')

for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    max_length = 1024
    stride = 512

    ppls = []
    for k in range(10):
       lls = []
       modelp = perturb_attn(copy.deepcopy(model1), p)
       for i in tqdm(range(0, encodings1.input_ids.size(1), stride)):
           begin_loc = max(i + stride - max_length, 0)
           end_loc = min(i + stride, encodings1.input_ids.size(1))
           trg_len = end_loc - i    # may be different from stride on last loop
           input_ids = encodings1.input_ids[:,begin_loc:end_loc].to('cuda')
           target_ids = input_ids.clone()
           target_ids[:,:-trg_len] = -100

           with torch.no_grad():
               outputs = modelp(input_ids, labels=target_ids)
               log_likelihood = outputs[0] * trg_len
           lls.append(log_likelihood)
       mean = torch.stack(lls).sum() / (end_loc)
       ppls.append(torch.exp(mean))
    print(p, ", Monolingual ppl: ", torch.mean(torch.stack(ppls)), " std ", torch.std(torch.stack(ppls)))

    ppls = []
    for k in range(10):
        lls = []
        modelp = perturb_attn(copy.deepcopy(model2), p)
        for i in tqdm(range(0, encodings2.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings2.input_ids.size(1))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = encodings2.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100

            with torch.no_grad():
                outputs = modelp(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        mean = torch.stack(lls).sum() / (end_loc)
        ppls.append(torch.exp(mean))
    print(p, ", Bilingual ppl: ", torch.mean(torch.stack(ppls)), " std ", torch.std(torch.stack(ppls)))

