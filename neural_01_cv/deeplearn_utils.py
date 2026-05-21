import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from torchinfo import summary

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# функция для расчета точности
def accuracy_fn(logps, labels):
    pred_classes = torch.argmax(torch.exp(logps), axis=1)
    val_classes = labels
    return float(torch.eq(pred_classes, val_classes).sum() / labels.shape[0])

# функция для обучения
def train(model, epochs=10000, report_positions=20):

    results = {'epoch_count': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # прогоняем данные по нейросети
    for epoch in range(epochs):

        # Put the model in training mode
        model.train()

        y_logps = model(X_train) #логарифмы вероятности отнесения к классам
        loss = criterion(y_logps, y_train) #кросс-энтропия

        acc = accuracy_fn(y_logps, y_train) # calculate the accuracy; convert the labels to integers

        optimizer.zero_grad() # reset the gradients so they don't accumulate each iteration
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass

        # Put the model in evaluation mode
        model.eval()

        with torch.inference_mode():
            y_val_logps = model(X_val)

            valid_loss = criterion(y_val_logps, y_val)
            valid_acc = accuracy_fn(y_val_logps, y_val)

        # Print progress a total of 20 times
        if epoch % (epochs // report_positions) == 0 or epochs<50:
            print(f'Epoch: {epoch+1:4.0f} | Train Loss: {loss:.5f}, Accuracy: {acc*100:.2f}% | \
                Validation Loss: {valid_loss:.5f}, Accuracy: {valid_acc*100:.2f}%')

            results['epoch_count'] += [epoch]
            results['train_loss'] += [loss.cpu().detach().numpy()]
            results['train_acc'] += [acc]
            results['val_loss'] += [valid_loss.cpu().detach().numpy()]
            results['val_acc'] += [valid_acc]

    return results

# рисовалка графиков
def plot_results(results):

    fig, axs = plt.subplots(1,2)

    fig.set_size_inches(10,3)

    for i, loss_acc in enumerate(['loss', 'acc']):
        for train_val in ['train', 'val']:
            axs[i].plot(results['epoch_count'], results[f'{train_val}_{loss_acc}'], label=f'{loss_acc} {train_val}')

        axs[i].legend()

    plt.show()