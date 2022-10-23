import hydra
import os
import jax
from flax import linen as nn
import jax.numpy as jnp
from typing import Callable
from tqdm import tqdm
import numpy as np
import math
from collections import defaultdict
# optimization
import optax
# visualizations
import matplotlib.pyplot as plt


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
class LinearNetwork(nn.Module):
  n: int
  num_tasks: int
  C_init: Callable = nn.initializers.lecun_normal()
  
  @nn.compact
  def __call__(self, x, task_ids):
    C = self.param('C', self.C_init, (self.num_tasks, self.n))
    return (C[task_ids] * x).sum(axis=1)

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()


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



@hydra.main(config_name='configs/linear')
def main(config):
    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()
    config = config['configs']
    print(config)
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
      
      model = LinearNetwork(arch_conf['n'], max_tasks)
      sample_input = jnp.ones((optim_conf['batch_size'], arch_conf['n'])), jnp.zeros((optim_conf['batch_size'],), dtype=jnp.int32)
      variables = model.init(random_key, *sample_input)
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
      # Learn linear classifier
      for epoch in range(optim_conf['epochs']):
          # retrieve loaders
          train_loaders = [x[0] for x in loaders]
          test_loaders = [x[1] for x in loaders]

          # collect metrics
          epoch_accuracies = []
          pbar = tqdm(range(len(train_loaders[0])))
          for samples in zip(*train_loaders):
            pbar.update(1)
            x, binary_labels, task_ids = prepare_samples(samples, classes)
            

            
            def loss_fn(variables):
              model_predictions = model.apply(variables, x, task_ids)
              class_predictions = (model_predictions >= 0.5)
              accuracy = jnp.array(class_predictions == binary_labels, dtype=jnp.int32).sum() / model_predictions.shape[0]
              
              loss = (0.5 * (model_predictions - binary_labels) ** 2).mean()
              loss += sum(l2_loss(w, alpha=optim_conf["alpha"]) for w in jax.tree_leaves(variables["params"]))
              return loss, accuracy

            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, accuracy), grads = grad_fn(variables)
            updates, opt_state = tx.update(grads, opt_state, params=variables)
            # grab predictions before updating
            predictions = model.apply(variables, x, task_ids)
            variables = optax.apply_updates(variables, updates)

            # project task vectors to make sure they have unit-norms
            C = variables['params']['C']
            variables = variables.copy({'params': {'C': C / jnp.linalg.norm(C, axis=1, keepdims=True)}})

            # metrics
            epoch_accuracies.append(accuracy)
            pbar.set_description(f'Epoch {epoch + 1} / {optim_conf["epochs"]}, Loss: {loss:.3f}, Acc: {accuracy:.3f}')
          
          fancy_log(f'Average training accuracy of epoch {epoch + 1} is {jnp.array(epoch_accuracies).mean():.3f}')
      
      C = variables['params']['C'][:T].T

      
      with open(f'C_{run_index + 1}.npy', 'wb') as f:
        np.save(f, np.array(C))
      
      fancy_log('SVD decomposition....')
      U, S, V_T = jnp.linalg.svd(C)
      U = U[:, :arch_conf['k']]
      S = jnp.diag(S[:arch_conf['k']])
      V_T = V_T[:arch_conf['k'], :]
      W = U.T
      G = S @ V_T
      fancy_log('Finished SVD decomposition')


      fancy_log('Starting evaluation on test dataset...')
      eval_mse = []
      for loader_index, loader in enumerate(test_loaders):
        task_eval_mse = []
        for x, y in loader:
            # prepare data
            x = x.numpy()
            x = x.reshape(x.shape[0], -1)
            y = y.numpy()
            binary_labels = (y == classes[loader_index][1]).astype(jnp.int32)
            task_ids = jnp.array([loader_index for _ in range(x.shape[0])])
            out = (G[:, task_ids] * (x @ W.T).T).sum(axis=0)
            curr_mse = (0.5 * (out - binary_labels)**2).mean()
            task_eval_mse.append(curr_mse)
        task_eval_mse = jnp.array(task_eval_mse).mean()
        collector.stats[run_index][f'Task {run_index} Eval MSE'].append(task_eval_mse)
        eval_mse.append(task_eval_mse)
      
      eval_mse = jnp.array(eval_mse).mean()
      collector.stats[run_index]['Eval MSE'].append(eval_mse)
      
      fancy_log('Finished evaluation on test dataset...')

      fancy_log('Computing slope...')
      collector.stats[run_index]['Slope'].append((S ** 2).sum() / T)
      fancy_log('Finished slope computation...')

      fancy_log('Measuring perf. under noise...')
      corruption_type = corruption_conf['type']
      for value in corruption_conf[corruption_type]['iters']:
        p_total_mse = []
        for loader_index, loader in enumerate(test_loaders):
          p_task_mse = []
          for x, y in loader:
            # prepare data
            x = x.numpy()
            x = x.reshape(x.shape[0], -1)
            y = y.numpy()
            binary_labels = (y == classes[loader_index][1]).astype(jnp.int32)
            task_ids = jnp.array([loader_index for _ in range(x.shape[0])])
            corruption_key, _ = jax.random.split(random_key)
            W_corr = corrupt_W(W, corruption_type, value, corruption_key)

            corr_out = (G[:, task_ids] * (x @ W_corr.T).T).sum(axis=0)
            predictions = model.apply(variables, x, task_ids)
            
            curr_p_task_mse = (0.5 * (corr_out - binary_labels)**2).mean()
            p_task_mse.append(curr_p_task_mse)
          
          p_task_mse = jnp.array(p_task_mse).mean()
          collector.stats[run_index][f'Task {loader_index + 1} Noisy MSE'].append(p_task_mse)
          p_total_mse.append(p_task_mse)

        p_total_mse = jnp.array(p_total_mse).mean()
        collector.stats[run_index]['Mean Noisy MSE'].append(p_total_mse)        
      fancy_log('Finished measuring perf. under noise! :)')

    runs_share_fig = ['Mean Noisy MSE']
    plot_as_p_changes = ['Mean Noisy MSE'] + [f'Task {i} Noisy MSE' for i in range(max_tasks)]

    max_key = len(task_conf['classes']) - 1
    for key in collector.stats[max_key].keys():
      fig = plt.figure()
      ax = fig.add_subplot(111)
      for run_index in collector.stats.keys():
        if key not in runs_share_fig:
          fig = plt.figure()
          ax = fig.add_subplot(111)
        x_axis = corruption_conf[corruption_type]['iters']
        x_label = 'Noise'
        label = f"Num tasks: {len(task_conf['classes'][run_index])}"
        if key in collector.stats[run_index] and key in plot_as_p_changes:
          collector.make_plot(fig, ax, run_index, key, label=label, x_axis=x_axis, x_label=x_label)

    plot_as_tasks_change = ['Eval MSE', 'Slope']
    # All keys
    for key in collector.stats[max_key].keys():
      fig = plt.figure()
      ax = fig.add_subplot(111)
      values = []
      # all runs
      for run_index in collector.stats.keys():
        # check if key is has values that change with number of tasks
        if key in plot_as_tasks_change:
          # if yes, add the value of the key for run with run_index
          values.append(collector.stats[run_index][key])
      if key in plot_as_tasks_change:
        x_axis = [arch_conf['k'] + x for x in range(max_key + 1)]
        x_label = 'Number of tasks'
        collector.make_plot(fig, ax, None, key, values=values, x_axis=x_axis, x_label=x_label)


if __name__ == '__main__':
    main()
