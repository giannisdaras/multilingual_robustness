import jax.numpy as jnp
import jax
from collections import defaultdict
import itertools
import matplotlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('ggplot')


def filter_classes(dataset, digits):
  new_dataset = []
  print('Filtering classes...')
  for tr in tqdm(dataset):
    if tr[1] in digits:
      new_dataset.append(tr)
  print('Finished filtering classes...')
  return new_dataset

def fancy_log(msg, symbol='*', reps=100):
  print(f'{symbol}' * reps)
  print(f'\t\t\t {msg} \t\t\t')
  print(f'{symbol}' * reps)

def flatten_list(ls):
  empty_list = []
  for x, y in ls:
    empty_list.append(x)
    empty_list.append(y)
  return tuple(set(empty_list))


def get_task_ids(labels, classes):
  # assumes no overlap in digits
  task_ids = []
  for y in labels:
    for id, elems in enumerate(classes):
      if y in elems:
        task_ids.append(id)
  return jnp.array(task_ids)

def get_binary_labels(labels, ids, classes):
  binary_labels = []
  for task_id, label in zip(ids, labels):
    task_classes = classes[task_id]
    if label == task_classes[0]:
      binary_labels.append(0)
    else:
      binary_labels.append(1)
  return jnp.array(binary_labels)

def corrupt_W(W, type, value, random_key):
  if type == 'random_deletions':
    p = 1 - value
    mask = jax.random.bernoulli(random_key, p=p, shape=W.shape)
    W = W * mask
  elif type == 'additive_noise':
    W = W + jax.random.normal(random_key, shape=W.shape) * value
  elif type == 'multiplicative_noise':
    W = W * (jax.random.normal(random_key, shape=W.shape) + 1) * value
  return W

class Marker:
  def __init__(self, marker, markersize, markerfacecolor, markeredgecolor):
    self.marker = marker
    self.markersize = markersize
    self.markerfacecolor = markerfacecolor
    self.markeredgecolor = markeredgecolor


class StatsCollector:
  stats = defaultdict(lambda: defaultdict(list))
  markers = itertools.cycle([
    Marker('*', 10, 'red', None),
    Marker('o', 10, 'pink', None),
    Marker('P', 10, 'blue', None),
    Marker('s', 10, 'black', None),
    Marker('v', 10, 'gray', None),
    Marker('^', 10, 'green', None),
    Marker('x', 10, 'yellow', None),
    Marker('D', 10, 'orange', None),
    Marker('2', 10, 'darkviolet', None),
    Marker('h', 10, 'brown', None)

    ])
  linestyles = itertools.cycle(['-.'])

  def __call__(self, predictions, binary_labels, loss, variables, run_id):
    W = variables['params']['W']
    gamma = variables['params']['G'][0]
    
    self.stats[run_id]['Accuracy'].append(((predictions > 0.5) == binary_labels).mean())
    self.stats[run_id]['Loss'].append(loss)
    self.stats[run_id]['Quadratic'].append(gamma.T @ (W @ W.T) @ gamma)

    # U, singular_values, _ = jnp.linalg.svd(W)
    # self.stats[run_id]['max_singular'].append(singular_values[0])
    # self.stats[run_id]['stable_rank'].append(jnp.sqrt((W**2).sum()) / singular_values[0])
  
  def make_plot(self, fig, ax, run_index, key, base_path='', label=None, x_label=None, x_axis=None, values=None):
    marker = next(self.markers)
    plot_args = {
      'marker': marker.marker, 
      'markersize': marker.markersize, 
      'markerfacecolor': marker.markerfacecolor, 
      'markeredgecolor': marker.markeredgecolor,
      'linestyle': next(self.linestyles),
      'label': label    
    }

    if values is None:
      values = self.stats[run_index][key]

    if x_axis is None:
      plt.plot(values, **plot_args)
      ax.set_xticklabels([])
    else:
      plt.plot(x_axis, values, **plot_args)
    ax.set_ylabel(key)
    if x_label is not None:
      ax.set_xlabel(x_label)
    plt.legend(fontsize=15, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, key + '.pdf'))
  

