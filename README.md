## Multitasking Models are Robust to Structural Failure: A Neural Model for Bilingual Cognitive Reserve

### Abstract
> We find a surprising connection between multitask learning and robustness to neuron failures. Our experiments show that bilingual language models retain higher performance under various neuron perturbations, such as random deletions, magnitude pruning and weight noise compared to equivalent monolingual ones. 
We provide a theoretical justification of this robustness by mathematically analyzing linear representation learning and showing that multitasking creates more robust representations. 
Our analysis connects robustness to spectral properties of the learned representation and proves that multitasking leads to higher robustness for diverse task vectors.


### Results

#### Monolingual vs. Bilingual GPT-2 Experiment: Model perplexity as a function of weight deletion
<img src="visuals/gpt2_random_delete.png" style="width:400px;"/>

#### Linear Representation Visual Model Experiments: MSE as a function of additive noise on model weights
<img src="visuals/cifar_noisy_mse.png" style="width:400px;"/>

##### CIFAR dataset

<img src="visuals/mnist_noisy_mse.png" style="width:400px;"/>

##### MNIST dataset


### What's here
The code hosted in this repository is the one we used to run all the experiments in the paper.

1) Multi-task Linear Classifier experimentations:
 - binary_classification.py

2) GPT2 monolingual and bilingual fine-tuning, and perturbations test:
 - train_bilingual_gpt2.py
 - test_gpt2.py

3) GPT2ForClassification training and test code for GLUE
 - run_glue_test.py

### Acknowledgments

The code for the NLP experiments is exclusively based on the HuggingFace `transformers` [library](https://github.com/huggingface/transformers).
We are very grateful to the authors of the library for their work.
