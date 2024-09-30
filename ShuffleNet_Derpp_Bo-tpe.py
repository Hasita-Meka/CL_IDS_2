import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision.models import shufflenet_v2_x0_5
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import optuna  # Add this import for Optuna

# Define constants
BATCH_SIZE = 1024
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = 1000  # Define buffer size
ALPHA = 0.5  # MSE loss weight
BETA = 1.0  # Cross-entropy loss weight

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use a single folder for the dataset
data_dir = 'EVSE'  # Path to the folder containing the images

# Load dataset
full_dataset = ImageFolder(data_dir, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(full_dataset))  # 80% for training
test_size = len(full_dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load ShuffleNet model
def create_model(learning_rate):
    model = shufflenet_v2_x0_5(pretrained=True)
    num_classes = len(full_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()

# Buffer class for experience replay
class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.inputs = []
        self.labels = []
        self.logits = []

    def add_data(self, inputs, labels, logits):
        # Add new data to the buffer
        for i in range(len(inputs)):
            if len(self.inputs) < self.buffer_size:
                self.inputs.append(inputs[i].cpu())
                self.labels.append(labels[i].cpu())
                self.logits.append(logits[i].cpu())
            else:
                # If the buffer is full, replace randomly
                idx = np.random.randint(0, self.buffer_size)
                self.inputs[idx] = inputs[i].cpu()
                self.labels[idx] = labels[i].cpu()
                self.logits[idx] = logits[i].cpu()

    def get_data(self, batch_size, transform, device):
        # Fetch random data from the buffer
        indices = np.random.choice(len(self.inputs), batch_size, replace=False)
        buffer_inputs = torch.stack([self.inputs[i] for i in indices]).to(device)
        buffer_labels = torch.tensor([self.labels[i] for i in indices]).to(device)
        buffer_logits = torch.tensor(np.stack([self.logits[i] for i in indices])).to(device)

        # Apply transform if needed (make sure transform is compatible with Tensors)
        return buffer_inputs, buffer_labels, buffer_logits

    def is_empty(self):
        # Check if buffer is empty
        return len(self.inputs) == 0

# Derpp model for continual learning
class Derpp:
    def __init__(self, model, buffer_size, alpha, beta):
        self.model = model
        self.buffer = Buffer(buffer_size)
        self.alpha = alpha
        self.beta = beta

    def observe(self, inputs, labels, not_aug_inputs, optimizer):
        # Forward pass for current data
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)  # Retain graph for further backward pass
        tot_loss = loss.item()

        # Experience Replay with buffer data
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(BATCH_SIZE, transform, DEVICE)
            buf_outputs = self.model(buf_inputs)

            # MSE loss on logits
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward(retain_graph=True)  # Retain graph here as well
            tot_loss += loss_mse.item()

            # Cross-entropy loss on labels
            loss_ce = self.beta * criterion(buf_outputs, buf_labels)
            loss_ce.backward()  # No need to retain graph after the last backward call
            tot_loss += loss_ce.item()

        optimizer.step()

        # Add data to buffer
        self.buffer.add_data(not_aug_inputs, labels, outputs.data)
        return tot_loss

# Function to train the model using Derpp
def train_model(model, train_loader, optimizer, num_epochs, derpp_model):
    model.train()
    train_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # DER++ observation step
            loss = derpp_model.observe(images, labels, images, optimizer)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, f1, precision, recall, auc_roc, cm

# Optimization function for Optuna
def objective(trial):
    # Hyperparameter optimization
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    model = create_model(learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate Derpp model
    derpp_model = Derpp(model=model, buffer_size=BUFFER_SIZE, alpha=ALPHA, beta=BETA)

    # Train the model using DERPP
    train_loss = train_model(model, train_loader, optimizer, NUM_EPOCHS, derpp_model)

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    # Use accuracy as the objective metric
    return accuracy

# Start the Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # You can increase the number of trials for better results

# Print the best hyperparameters found
print("Best hyperparameters: ", study.best_params)

# After hyperparameter tuning, you can retrain the model with the best hyperparameters if desired
best_learning_rate = study.best_params['learning_rate']
best_model = create_model(best_learning_rate)
best_optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)

# Retrain with the best parameters
derpp_model = Derpp(model=best_model, buffer_size=BUFFER_SIZE, alpha=ALPHA, beta=BETA)
train_loss = train_model(best_model, train_loader, best_optimizer, NUM_EPOCHS, derpp_model)

# Evaluate the best model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(best_model, test_loader)

# Print performance metrics for the best model
print(f'Average Loss: {avg_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print('Confusion Matrix:\n', cm)
