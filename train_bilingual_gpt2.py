import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import logging
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import math
from datasets import load_dataset
from argparse import ArgumentParser
from itertools import chain, cycle
from collections import OrderedDict
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from sklearn.model_selection import train_test_split


class MpLogger:
    def __init__(self, logger, rank):
        self.logger = logger
        self.rank = rank

    def info(self, message):
        if self.rank == 0:
            self.logger.info(message)


logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


def configure_optimizer(model):
    for k, v in model.named_parameters():
        v.requires_grad=True
    optim_groups = [
        {"params": model.parameters(), "weight_decay": 0.01}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, amsgrad=True)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=150, num_training_steps=128867)
    return optimizer, scheduler


def mp_setup(rank, world_size):
    global_rank = int(os.environ['SLURM_NODEID']) * torch.cuda.device_count() + rank
    if os.getenv("MASTER_ADDR") is None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size, init_method='env://')
    torch.cuda.set_device(rank)
    return global_rank


def mp_cleanup():
    dist.destroy_process_group()


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


def mp_run(rank, args, project_dir, working_dir):
    global_rank = mp_setup(rank, args.world_size) if args.multiprocessing else rank
    if global_rank == 0:
        print(f'Working directory: {working_dir}')
        print('Running with the following config')

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    logger = MpLogger(logger, global_rank)
    logger.info(f'Working directory: {working_dir}')
    logger.info(f'Project dir: {project_dir}')
    logger.info(args)

    block_size = args.block_size
    batch_size = args.batch_size

    torch.manual_seed(42)
    np.random.seed(42)

    # loading tokenizer from the saved model path
    tokenizer = GPT2Tokenizer.from_pretrained(working_dir+'/token_gpt2')
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    raw_dataset = load_dataset("text", data_files=args.train_file, cache_dir=working_dir+"/cache_en")
    #raw_dataset = load_dataset("lambada", cache_dir=working_dir+"/cache_en")
    #raw_dataset = load_dataset('code_search_net', 'python', cache_dir=working_dir+"/cache_en")
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    model.to('cuda')
    if args.multiprocessing:
        model = DDP(model, device_ids=[rank])
    model.to(rank)

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    column_names = raw_dataset['train'].column_names
    text_column_name = "text"
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if block_size > tokenizer.model_max_length:
        logger.warning(
            f"Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    #model = load_dict(model, './monolingual_openai_1.pth')
    dataset = lm_datasets['train'].train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    optimizer, scheduler = configure_optimizer(model)
    train_sampler = DistributedSampler(train_dataset, rank=global_rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                            drop_last=True,
                            num_workers=1,
                            batch_size=batch_size,
                            sampler=train_sampler
                            )
    model.to(rank)
    model.train()
    loss = 0
    scheduler.step()
    for epoch in tqdm(range(args.n_epochs), disable=(global_rank != 0)):
        train_epoch_index = 0
        train_epoch_avg_loss = 0
        train_epoch_avg_ppls = 0
        scheduler.step()
        if global_rank == 0:
            logger.info(f'Running epoch {epoch + 1}/{args.n_epochs}')
        tqdm_sample = tqdm(train_dataloader, disable=(global_rank != 0))
        it = 0
        for x in tqdm_sample:
            attention_mask = torch.stack(x['attention_mask'], dim=1).to(rank)
            input_ids = torch.stack(x['input_ids'], dim=1).to(rank)
            labels = torch.stack(x['labels'], dim=1).to(rank)
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = output['loss'], output['logits']
            logger.info(loss)

            loss = loss.mean()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            train_epoch_index += 1
            train_epoch_avg_loss += loss.item()
            train_epoch_avg_ppls += torch.exp(loss).item()
            tqdm_sample.set_description(f'Current loss (normalized): {loss.item() :.5f}')

            if (global_rank == 0) and (it % 500 == 0):
                file_name = f'monolingual_openai_{epoch}_iter_{it+1}.pth'
                torch.save(model.state_dict(), file_name)
            it += 1

            if global_rank == 0:
                logger.info('avg_train_epoch_loss'+ str(train_epoch_avg_loss/train_epoch_index) + ' average ppls: ' + str(train_epoch_avg_ppls/train_epoch_index))
            
            if global_rank == 0:
                file_name = f'monolingual_openai_{epoch}.pth'
                logger.info(f'Saving intermediate checkpoint: {os.path.join(working_dir, file_name)}')
                torch.save(model.state_dict(), file_name)

        mp_cleanup()



def main(config):
    working_dir = os.getcwd()
    project_dir = os.getcwd()

    if  config.multiprocessing:
        mp.spawn(mp_run,
         args=(config, project_dir, working_dir),
         nprocs=torch.cuda.device_count(),
         join=True)
    else:
        mp_run(0, config, project_dir, working_dir)


if __name__ == '__main__':
    def get_process_log_level():
        log_level_main_node = logging.INFO
        log_level_replica_node = logging.WARNING
        return log_level_main_node

    parser = ArgumentParser()
    parser.add_argument("-mp", "--multiprocessing", default=False, action='store_true')
    parser.add_argument("--world_size", default=4)
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--n_epochs", default=4)
    parser.add_argument("--train_file", default="en_text.txt")
    parser.add_argument("--block_size", default=1024)
    parser.add_argument("--model_type", default="gpt2")
    parser.add_argument("--get_process_log_level", default=get_process_log_level())
    args = parser.parse_args()
    main(args)