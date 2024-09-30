
import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision.models import squeezenet1_1  # Import SqueezeNet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import optuna  # Import Optuna for HPO
from torchvision import datasets

# Define constants
NUM_EPOCHS = 30
BUFFER_SIZE = 500  # Size of the buffer
ALPHA = 0.1  # Weight for MSE loss
BETA = 0.1   # Weight for cross-entropy loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024  # Fixed batch size
TRAIN_SPLIT = 0.6  # 80% of data for training, 20% for testing

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset from a single folder
data_dir = 'EVSE'  # Single folder for all data
full_data = ImageFolder(data_dir, transform=transform)

# Split dataset into train and test sets
train_size = int(TRAIN_SPLIT * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size])

# Define a Buffer class for storing inputs, labels, and logits
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

    def get_data(self, batch_size, device):
        # Fetch random data from the buffer
        indices = np.random.choice(len(self.inputs), batch_size, replace=False)
        buffer_inputs = torch.stack([self.inputs[i] for i in indices]).to(device)
        buffer_labels = torch.tensor([self.labels[i] for i in indices]).to(device)
        buffer_logits = torch.tensor(np.stack([self.logits[i] for i in indices])).to(device)

        return buffer_inputs, buffer_labels, buffer_logits

    def is_empty(self):
        # Check if buffer is empty
        return len(self.inputs) == 0

# Define Derpp class
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
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(BATCH_SIZE, DEVICE)
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

# Load SqueezeNet model
def get_squeezenet():
    model = squeezenet1_1(weights="DEFAULT")  # Use the latest weights
    num_classes = len(full_data.classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))  # Modify the final layer
    model.num_classes = num_classes
    return model.to(DEVICE)

# Define the objective function for Optuna
def objective(trial):
    global criterion  # Make criterion global
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    # Create DataLoader with the fixed batch size
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create SqueezeNet model
    squeezenet_model = get_squeezenet()
    derpp_model = Derpp(squeezenet_model, BUFFER_SIZE, ALPHA, BETA)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(derpp_model.model.parameters(), lr=learning_rate)

    # Train the Derpp model
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            not_aug_inputs = images.clone()  # For demonstration, we use the same images

            # Observe and update model
            loss = derpp_model.observe(images, labels, not_aug_inputs, optimizer)
            epoch_loss += loss

    avg_loss = epoch_loss / len(train_loader)

    # Return the average loss as the objective value
    return avg_loss

# Create a study and optimize hyperparameters
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)  # You can change the number of trials as needed

# Print the best trial results
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value:.4f}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Load the best parameters to train the model
best_learning_rate = trial.params['learning_rate']

# Load final model with best hyperparameters
final_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
final_test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Final training with best parameters
squeezenet_model = get_squeezenet()
derpp_model = Derpp(squeezenet_model, BUFFER_SIZE, ALPHA, BETA)
optimizer = optim.Adam(derpp_model.model.parameters(), lr=best_learning_rate)

# Train the model with best hyperparameters
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for images, labels in final_train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        not_aug_inputs = images.clone()

        loss = derpp_model.observe(images, labels, not_aug_inputs, optimizer)
        epoch_loss += loss

    avg_loss = epoch_loss / len(final_train_loader)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

# Evaluate the final model
def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(test_loader), all_labels

avg_loss, all_labels = evaluate(derpp_model, final_test_loader)

# End timing
end_time = time.time()
time_taken = end_time - start_time

# Calculate metrics (Assuming `predictions` are obtained from the model's outputs)
predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # Get predicted classes

# Calculate metrics
accuracy = accuracy_score(all_labels, predictions)
f1 = f1_score(all_labels, predictions, average='weighted')
precision = precision_score(all_labels, predictions, average='weighted')
recall = recall_score(all_labels, predictions, average='weighted')
auc_roc = roc_auc_score(all_labels, predictions, multi_class='ovr')

# Print evaluation results
print(f"Test Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Training Time: {time_taken:.2f} seconds")

