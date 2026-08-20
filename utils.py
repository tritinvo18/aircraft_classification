import random
import os
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

# Download DinoV3 from GitHub if missing
if not os.path.exists('dinov3'):
    print("dinov3 folder not found. Cloning from GitHub...")
    subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dinov3.git'], check=False)

# Set the seed across all processes to ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False

def plot_class_distribution(*dataset, dataset_names=None, figsize=(14, 6)):
    """
    Plots the class distribution for one or more datasets.
    Parameters:
    - dataset: One or more datasets (e.g., trainval_dataset, test_dataset).
    - dataset_names: List of names corresponding to each dataset for labeling.
    - figsize: Tuple specifying the size of the figure.
    """
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(len(dataset))]

    # Collect data from all datasets
    df_list = []
    for ds, name in zip(dataset, dataset_names):
        # Handle both ImageFolder and Subset datasets
        if hasattr(ds, 'samples'):
            # ImageFolder datasets have a 'targets' attribute that contains the labels
            labels_full = [label for _, label in ds.samples]
            class_names = ds.classes
        else:
            # Subset datasets - get labels from indices
            labels_full = [ds.dataset.targets[i] for i in ds.indices]
            class_names = ds.dataset.classes
        df = pd.DataFrame({
            'class': [class_names[label] for label in labels_full],
            'split': name,
            'count': 1
        })
        df_list.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(df_list, ignore_index=True)
    # Group by class and split, then count occurrences
    class_distribution = combined_df.groupby(['class', 'split']).count().reset_index()

    # Create grouped bar plot
    plt.figure(figsize=figsize)
    sns.barplot(data=class_distribution, x='class', y='count', hue='split')
    plt.title('Class Distribution Across Datasets')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.legend(title='Split')
    plt.tight_layout()
    plt.show()

def setup_model(model, num_classes, freeze_backbone=False, device="cpu"):
    # Adapt the architecture for the new number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # If needed, freeze the backbone layers to prevent them from being updated during training
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Only the final layer will be trainable
        for param in model.fc.parameters():
            param.requires_grad = True
    return model.to(device)

def train_epoch(model, dataloader, criterion, optimizer, epoch, device):

    # Put the model in train mode
    model.train()

    # For all batches in the training dataset
    train_loss, correct, total = [], 0.0, 0.0
    
    for inputs, labels in dataloader:
        # Get the inputs and labels from the dataloader and move to device (GPU or CPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero-out the gradient for this batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss, gradients with backpropagation, and update the weights
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy stats
        train_loss += [loss.cpu().item()]
        predicted = torch.argmax(outputs, dim=1)
        correct += torch.sum(predicted == labels).cpu().item()
        total += len(labels)

    # Average stats for the epoch
    mean_train_loss = sum(train_loss) / len(train_loss)
    train_accuracy = correct / total

    return mean_train_loss, train_accuracy

def eval_epoch(model, dataloader, criterion, epoch, device):

    # Put the model in "eval" mode
    model.eval()
    all_labels = []
    all_preds = []

    # Validation loop: for all batches in the validation dataset
    with torch.no_grad(): # Not build the computation graph for backpropagation, and thus no gradients will be computed or stored for the tensors involved in those operations
        val_loss, val_total = 0.0, 0.0
        for inputs, labels in dataloader:
            # Get the inputs and labels from the dataloader and move to device (GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            # Calculate the loss and accuracy at the validation split
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_total += inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    mean_val_loss = val_loss / max(val_total, 1)
    val_accuracy = balanced_accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

    return mean_val_loss, val_accuracy