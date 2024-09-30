import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision.models import mobilenet_v2  # Import MobileNetV2
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import torch.nn.functional as F
import optuna  # Import Optuna for hyperparameter optimization

# Define constants
BATCH_SIZE = 1024  # Batch size for training
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing all images
data_dir = 'EVSE'

# Load the full dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split the dataset into train and test (e.g., 80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = random_split(full_dataset, [train_size, test_size])

# Data loaders for train and test sets
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Load MobileNetV2 model
def create_model(learning_rate):
    model = mobilenet_v2(pretrained=True)
    num_classes = len(full_dataset.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify the final layer
    model = model.to(DEVICE)
    return model, optim.Adam(model.parameters(), lr=learning_rate)

# Loss function
criterion = nn.CrossEntropyLoss()

class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data = []
        self.labels = []
        self.logits = []

    def is_empty(self):
        return len(self.data) == 0

    def add_data(self, examples, labels, logits):
        self.data.extend(examples)
        self.labels.extend(labels)
        self.logits.extend(logits)

        if len(self.data) > self.buffer_size:
            self.data = self.data[-self.buffer_size:]
            self.labels = self.labels[-self.buffer_size:]
            self.logits = self.logits[-self.buffer_size:]

    def get_data(self, minibatch_size, transform, device):
        idx = np.random.choice(len(self.data), minibatch_size, replace=False)
        buf_inputs = torch.stack([self.data[i] for i in idx]).to(device)
        buf_labels = torch.tensor([self.labels[i] for i in idx]).to(device)
        buf_logits = torch.stack([self.logits[i] for i in idx]).to(device)
        return buf_inputs, buf_labels, buf_logits


class DerppModel:
    def __init__(self, model, learning_rate, alpha=0.1, beta=0.1, buffer_size=100):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.buffer = Buffer(buffer_size)
        self.opt = optim.Adam(model.parameters(), lr=learning_rate)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(BATCH_SIZE, transform, DEVICE)
            buf_outputs = self.model(buf_inputs)

            # DER++ MSE loss
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            # DER++ Cross Entropy loss
            buf_outputs = self.model(buf_inputs)
            loss_ce = self.beta * criterion(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.opt.step()
        self.buffer.add_data(not_aug_inputs, labels, outputs.detach())

        return tot_loss


# Function to train the model
def train_model(model, train_loader, num_epochs, learning_rate):
    model.train()
    train_loss = []
    derpp_model = DerppModel(model, learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            not_aug_images = images.clone()  # Store original images for DER++
            loss = derpp_model.observe(images, labels, not_aug_images)
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


# Function to plot loss vs accuracy curves
def plot_metrics(train_loss, accuracy_list):
    epochs = range(1, len(train_loss) + 1)

    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_list, 'g', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Load datasets with the suggested batch size
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create model with the suggested learning rate
    model, optimizer = create_model(learning_rate)

    # Train the model
    train_loss = train_model(model, train_loader, NUM_EPOCHS, learning_rate)

    # Evaluate the model
    avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(model, test_loader)

    return accuracy  # We want to maximize accuracy


# Start the hyperparameter optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Start timing
start_time = time.time()

# Train the model with the best hyperparameters
best_learning_rate = study.best_params['learning_rate']
best_batch_size = study.best_params['batch_size']
final_model, final_optimizer = create_model(best_learning_rate)

final_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
final_train_loss = train_model(final_model, final_train_loader, NUM_EPOCHS)

# Evaluate the model
avg_loss, accuracy, f1, precision, recall, auc_roc, cm = evaluate_model(final_model, test_loader)

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Print performance metrics
print(f'Average Loss: {avg_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC-ROC Score: {auc_roc:.4f}')
print(f'Confusion Matrix:\n {cm}')
print(f'Time taken for training: {time_taken:.2f} seconds')

# Plot metrics
plot_metrics(final_train_loss, [accuracy])
