# dataloading
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.datasets import fetch_20newsgroups
# utilities
from utils import filter_classes
import transformers
import os
from tqdm import tqdm

def load_ImageNet(batch_size, classes, max_samples=1000000, imgnet_dir='/raid/imgnet/ILSVRC/Data/CLS-LOC/'):
    train_dataset = CustomImageNet(classes, is_train=True, max_samples=max_samples)
    test_dataset = CustomImageNet(classes, is_train=False, max_samples=max_samples)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True, num_workers=1)
    return train_loader, test_loader


def load_MNIST(batch_size, classes, max_samples=10000000):
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))


    test_dataset = datasets.MNIST('../data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    train_dataset = filter_classes(train_dataset, classes)
    test_dataset = filter_classes(test_dataset, classes)
    
    train_dataset = train_dataset[:max_samples]
    test_dataset = test_dataset[:max_samples]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    return train_loader, test_loader


def load_CIFAR10(batch_size, classes, max_samples=10000000):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    train_dataset = filter_classes(train_dataset, classes)
    test_dataset = filter_classes(test_dataset, classes)
    
    train_dataset = train_dataset[:max_samples]
    test_dataset = test_dataset[:max_samples]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    return train_loader, test_loader


class CustomImageNet(Dataset):
    def __init__(self, classes, is_train=True, max_samples=10000, imgnet_dir='/raid/imgnet/ILSVRC/Data/CLS-LOC/'):
        self.classes = classes
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transformations = [transforms.ToTensor(), transforms.Resize((224, 224)), normalize]


        train_dir = os.path.join(imgnet_dir, 'train')
        test_dir = os.path.join(imgnet_dir, 'val')

        if is_train:
            self.dataset = datasets.ImageFolder(train_dir, transforms.Compose(transformations))
        else:
            self.dataset = datasets.ImageFolder(test_dir, transforms.Compose(transformations))
        print("Selecting valid indices...")
        self.valid_indices = []
        for index, y in enumerate(tqdm(self.dataset.targets)):
            if y in classes:
                self.valid_indices.append(index)
            if len(self.valid_indices) > max_samples:
                break
        print("Finished selecting valid indices....")

        resnet50 = models.resnet50(pretrained=True)
      
        modules = list(resnet50.children())[:-1]
        self.encoder = nn.Sequential(*modules).cuda()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def __getitem__(self, valid_idx):
        idx = self.valid_indices[valid_idx]
        encoded = self.encoder(self.dataset[idx][0].unsqueeze(0))[0]
        return encoded, self.dataset[idx][1]

    def __len__(self):
        return len(self.valid_indices)

        
class Newsgroup(Dataset):
    def __init__(self, classes, is_train=True, max_samples=100000):
        self.classes = classes
        if is_train:
            dataset = []
            for x, y in zip(
                fetch_20newsgroups(subset='train').data, 
                fetch_20newsgroups(subset='train').target):
                if y in classes:
                    dataset.append((x, y))
            self.dataset = dataset[:max_samples]

        else:
            dataset = []
            for x, y in zip(
                fetch_20newsgroups(subset='test').data, 
                fetch_20newsgroups(subset='test').target):
                if y in classes:
                    dataset.append((x, y))
            self.dataset = dataset[:max_samples]
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased').to('cuda:1')
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_text(self, text, max_context=64):
        devices = np.array(['cuda:1', 'cuda:2', 'cuda:3'])
        device = np.random.choice(devices)
        self.bert = self.bert.to(device)
        with torch.no_grad():
            encoded_text = self.tokenizer(text, return_tensors='pt')
            # limit context longer than max_context
            encoded_text['input_ids'] = encoded_text['input_ids'][:, :max_context].to(device)
            encoded_text['token_type_ids'] = encoded_text['token_type_ids'][:, :max_context].to(device)
            encoded_text['attention_mask'] = encoded_text['attention_mask'][:, :max_context].to(device)

            # get pooled output
            output = self.bert(**encoded_text)[-1].squeeze().cpu()
        return output
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.encode_text(self.dataset[idx][0]), self.dataset[idx][1]


def load_Newsgroup20(batch_size, classes, max_samples=100000):
    train_dataset = Newsgroup(classes, is_train=True, max_samples=max_samples)
    test_dataset = Newsgroup(classes, is_train=False, max_samples=max_samples)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_ImageNet(8, [0, 1, 2])
