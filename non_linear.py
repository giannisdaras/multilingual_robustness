import hydra
import os
import jax
from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Any
from tqdm import tqdm
import numpy as np
import math
from collections import defaultdict
# optimization
import optax
# visualizations
import matplotlib.pyplot as plt
from jax_resnet import pretrained_resnet
import functools

# utilies
from utils import flatten_list
from utils import fancy_log
from utils import filter_classes
from utils import get_binary_labels
from utils import get_task_ids
from utils import StatsCollector
from utils import corrupt_W

# dataloading
from dataloaders import load_MNIST, load_CIFAR10, load_Newsgroup20, load_ImageNet


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_binary_cross_entropy(logits, labels):
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  return (-labels * log_p - (1. - labels) * log_not_p).mean()


class ResnetWithHead(nn.Module):
  base_model: Any
  input_dim: int
  num_tasks: int
  head_init: Callable = nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, x, task_ids):
    representation = self.base_model(x)
    C = self.param('C', self.head_init, (self.num_tasks, self.input_dim))
    return (C[task_ids] * representation).sum(axis=1)


def prepare_samples(samples, classes):
    num_loaders = len(samples)
    batch_size = samples[0][0].shape[0]  
    # here we have a sample from each loader
    x_samples = []
    y_samples = []
    ids = []
    for loader_index in range(num_loaders):
      ids += [loader_index for _ in range(batch_size)]
      x_sample = samples[loader_index][0].numpy()
      y_sample = samples[loader_index][1].numpy()
      # convert to binary
      y_sample = (y_sample == classes[loader_index][1]).astype(jnp.int32)

      x_samples.append(x_sample)
      y_samples.append(y_sample)

    # concatenate samples from all loaders
    x = jnp.array(x_samples)
    x = jnp.reshape(x, (-1,) + x.shape[2:]).reshape(batch_size * num_loaders, -1)
    y = jnp.array(y_samples)
    binary_labels = jnp.reshape(y, (-1,) + y.shape[2:])
    ids = jnp.array(ids)
    return x, binary_labels, ids



@hydra.main(config_name='configs/non_linear')
def main(config):
    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()
    config = config['configs']
    optim_conf = config['optim']
    arch_conf = config['arch']
    task_conf = config['task']
    corruption_conf = config['corruption']

    random_key = jax.random.PRNGKey(config['seed'])
    collector = StatsCollector()

    for run_index, classes in enumerate(task_conf['classes']):
      max_tasks = max([len(_) for _ in task_conf['classes']])
      T = len(classes)
      print(f'Running for {T} tasks...')

      # get resnet architecture and pre-trained model    
      ResNeSt50, base_variables = pretrained_resnet(18)
      base_model = ResNeSt50()
      # remove final layer
      base_model = nn.Sequential(base_model.layers[:-1])
      model = ResnetWithHead(base_model=base_model, num_tasks=max_tasks, input_dim=512)
      sample_input = jnp.ones((1, config.data.size, config.data.size, config.data.num_channels)), jnp.zeros((optim_conf['batch_size'],), dtype=jnp.int32)
      # init all the model
      variables = model.init(random_key, *sample_input)
      
      variables = variables.unfreeze()
      # use pre-trained params for base_model
      variables["params"]["base_model"] = base_variables["params"]

      tx = optax.adamw(optim_conf['lr'], 
        weight_decay=optim_conf['weight_decay'])    
      opt_state = tx.init(variables)

      # get one loader per task
      num_loaders = len(classes)
      loaders = []
      for loader_index in range(num_loaders):
        # make sure that multitask models don't use more data
        max_samples = config['data']['max_samples'] // num_loaders
        print(f'Model uses: {max_samples * num_loaders} samples')
        if config.data.dataset_name == 'MNIST':
          loaders.append(load_MNIST(
            optim_conf['batch_size'] // num_loaders, 
            classes[loader_index], 
            max_samples=max_samples))
        elif config.data.dataset_name == 'CIFAR10':
          loaders.append(load_CIFAR10(
            optim_conf['batch_size'] // num_loaders, 
            classes[loader_index], 
            max_samples=max_samples))
        elif config.data.dataset_name == 'Newsgroup20':
            loaders.append(load_Newsgroup20(
            optim_conf['batch_size'] // num_loaders, 
            classes[loader_index], 
            max_samples=max_samples))  
        elif config.data.dataset_name == 'ImageNet':
            loaders.append(load_ImageNet(
            optim_conf['batch_size'] // num_loaders, 
            classes[loader_index], 
            max_samples=max_samples))
        else:
            raise ValueError("Uknown dataset name")
      
      if config.train:
        fancy_log("Starting finetuning....")
        # Learn linear classifier
        for epoch in range(optim_conf['epochs']):
            # retrieve loaders
            train_loaders = [x[0] for x in loaders]

            # collect metrics
            epoch_accuracies = []
            pbar = tqdm(range(len(train_loaders[0])))
            for samples in zip(*train_loaders):
              pbar.update(1)
              _, binary_labels, task_ids = prepare_samples(samples, classes)
              x = samples[0][0]            
              x = x.numpy().transpose(0, 2, 3, 1)

              @jax.jit
              def loss_fn(variables):
                model_predictions = model.apply(variables, x, task_ids)
                class_predictions = (sigmoid(model_predictions) >= 0.5)
                accuracy = jnp.array(class_predictions == binary_labels, dtype=jnp.int32).sum() / model_predictions.shape[0]
                return sigmoid_binary_cross_entropy(model_predictions, binary_labels), accuracy

              
              grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
              (loss, accuracy), grads = grad_fn(variables)
              updates, opt_state = tx.update(grads, opt_state, params=variables)
              variables = optax.apply_updates(variables, updates)
              epoch_accuracies.append(accuracy)
              pbar.set_description(f'Epoch {epoch + 1} / {optim_conf["epochs"]}, Loss: {loss:.3f}, Acc: {accuracy:.3f}')

      fancy_log('Starting evaluation on test dataset...')
      corruption_type = corruption_conf['type']
      for p in corruption_conf[corruption_type]['iters']:
        fancy_log(f"Deletion prob: {p}")
        eval_losses = []
        eval_accs = []        
        test_loaders = [x[1] for x in loaders]
        for loader_index, loader in enumerate(test_loaders):
          task_eval_losses = []
          task_eval_accs = []
          for x, y in tqdm(loader):
              # prepare data
              x = x.numpy().transpose(0, 2, 3, 1)
              y = y.numpy()
              binary_labels = (y == classes[loader_index][1]).astype(jnp.int32)
              task_ids = jnp.array([loader_index for _ in range(x.shape[0])])

              def corrupt(param, p=0.1):
                mask = jax.random.bernoulli(random_key, p=p, shape=param.shape)
                return (1 - mask) * param

              corr_variables = jax.tree_util.tree_map(functools.partial(corrupt, p=p), variables)

              def loss_fn(variables):
                model_predictions = model.apply(variables, x, task_ids)
                class_predictions = (sigmoid(model_predictions) >= 0.5)
                accuracy = jnp.array(class_predictions == binary_labels, dtype=jnp.int32).sum() / model_predictions.shape[0]
                return sigmoid_binary_cross_entropy(model_predictions, binary_labels), accuracy

              eval_loss, eval_acc = loss_fn(corr_variables)
              task_eval_losses.append(eval_loss)
              task_eval_accs.append(eval_acc)
              
          task_eval_losses = jnp.array(task_eval_losses)
          task_eval_accs = jnp.array(task_eval_accs)
          eval_losses.append(task_eval_losses.mean())
          eval_accs.append(task_eval_accs.mean())
          fancy_log(f"Task {loader_index}. Avg acc: {task_eval_accs.mean()} +/- {task_eval_accs.std()}")



if __name__ == '__main__':
    main()
