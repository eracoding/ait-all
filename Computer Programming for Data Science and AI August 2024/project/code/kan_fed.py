# 1. Imports
import os
import collections
from collections import defaultdict
import random
import joblib

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset

import torchvision.transforms as transforms
from kan import KAN

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class Server:
    def __init__(self, width, grid, k, seed, device, local_weight_ave=True):
        self.device = device
        self.global_model = KAN(width=width, grid=grid, k=k, seed=seed, device=device).to(device)
        self.send_funcs = {
            True: self.send_with_local_ave,
            False: self.send_without_local_ave
        }
        self.send = self.send_funcs[local_weight_ave]

    def send_without_local_ave(self, client):
        client.model.load_state_dict(self.global_model.state_dict())
        print(f"Sent model to Client {client.id}")

    def send_with_local_ave(self, client):
        global_params = self.global_model.state_dict()
        local_params = client.get_model_params()
        averaged_params = {key: (global_params[key] + local_params[key]) / 2 for key in global_params.keys()}
        client.model.load_state_dict(averaged_params)
        print(f"Sent model to Client {client.id}")

    def aggregate_weights(self, client_weights):
        avg_params = {key: torch.zeros_like(param) for key, param in client_weights[0].items()}
        for params in client_weights:
            for key in params.keys():
                avg_params[key] += params[key] / len(client_weights)
        return avg_params


class Client:
    def __init__(self, width, grid, k, seed, device, data, id):
        self.device = device
        self.model = KAN(width=width, grid=grid, k=k, seed=seed, device=device).to(device)
        self.data = data
        self.id = id

    def get_model_params(self):
        return self.model.state_dict()

    def local_train(self, optimizer, loss_fn, num_epochs, batch_size=64, lr=0.01):
        train_loader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        self.model.train()
        optimizer = optimizer(params=self.model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Client {self.id} Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")


# Define the server and clients
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = [10, 5, 2]  # Example KAN architecture
grid = 5
k = 3
seed = 42

server = Server(width=width, grid=grid, k=k, seed=seed, device=device)

client_names = ['Client 1', 'Client 2', 'Client 3', 'Client 4']
data_weights = [0.25, 0.25, 0.25, 0.25]
clients = [
    Client(width=width, grid=grid, k=k, seed=seed, device=device, data=subset, id=name)
    for subset, name in zip(split_dataset(train_dataset, data_weights), client_names)
]

# Federated training
num_rounds = 5
num_epochs = 5
lr = 0.001

for round_num in range(num_rounds):
    print(f"--- Round {round_num+1} ---")
    client_weights = []
    for client in clients:
        server.send(client)
        optimizer = torch.optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()
        client.local_train(optimizer, loss_fn, num_epochs=num_epochs, lr=lr)
        client_weights.append(client.get_model_params())
    aggregated_params = server.aggregate_weights(client_weights)
    server.global_model.load_state_dict(aggregated_params)


# Testing the global model
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
server.global_model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = server.global_model(data)
        test_loss += torch.nn.CrossEntropyLoss()(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

test_loss /= len(test_loader)
accuracy = 100. * correct / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
