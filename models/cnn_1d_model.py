"""
1D CNN Model for Sequential Classification
Member 4 Implementation - Formative Assignment 2

This module implements a 1D Convolutional Neural Network for sequential signal classification.
The model is designed to capture local temporal patterns in time-series data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for sequence classification.
    
    Architecture:
    - Convolutional layers with progressively increasing receptive field
    - Batch normalization for training stability
    - Dropout for regularization
    - Global average pooling to reduce dimensionality
    - Fully connected layers for classification
    
    Args:
        input_size (int): Number of features per timestep (e.g., 40 for MFCC)
        num_classes (int): Number of output classes
        num_filters (int): Number of filters in first conv layer (default: 32)
        dropout_rate (float): Dropout probability (default: 0.3)
        kernel_size (int): Size of convolutional kernel (default: 3)
    """
    
    def __init__(self, input_size, num_classes, num_filters=32, dropout_rate=0.3, kernel_size=3):
        super(CNN1D, self).__init__()
        
        self.num_filters = num_filters
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Convolutional Block 1: extract low-level features
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2  # 'same' padding
        )
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Convolutional Block 2: combine adjacent features
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Convolutional Block 3: high-level pattern recognition
        self.conv3 = nn.Conv1d(
            in_channels=num_filters * 2,
            out_channels=num_filters * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        fc_input_size = num_filters * 4
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)  # (batch, num_filters, seq_len)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch, num_filters, seq_len/2)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)  # (batch, num_filters*2, seq_len/2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch, num_filters*2, seq_len/4)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)  # (batch, num_filters*4, seq_len/4)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # (batch, num_filters*4, 1)
        x = self.dropout3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, num_filters*4)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch, 128)
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)  # (batch, num_classes)
        
        return x


class CNN1DTrainer:
    """
    Trainer class for 1D CNN model.
    Handles training, validation, and evaluation.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): CNN1D model instance
            device (str): 'cpu' or 'cuda'
            learning_rate (float): Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """
        Train the model.
        
        Args:
            train_loader: PyTorch DataLoader for training data
            val_loader: PyTorch DataLoader for validation data
            epochs (int): Number of training epochs
            patience (int): Early stopping patience
        
        Returns:
            dict: Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_cnn_model.pt')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cnn_model.pt'))
        return self.history
    
    def evaluate(self, test_loader, num_classes):
        """
        Evaluate on test set and return metrics.
        
        Returns:
            dict: Evaluation metrics and predictions
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'targets': all_targets,
            'report': classification_report(all_targets, all_predictions)
        }
    
    def plot_learning_curves(self):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Learning Curves - Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.title('Learning Curves - Accuracy')
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, conf_matrix):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - 1D CNN Model')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# 1. Load your data
X_train = np.load('train.npz')['arr_0']  # Shape: (num_samples, sequence_length, num_features)
y_train = np.load('train_labels.npy')

X_val = np.load('val.npz')['arr_0']
y_val = np.load('val_labels.npy')

X_test = np.load('test.npz')['arr_0']
y_test = np.load('test_labels.npy')

# 2. Create PyTorch datasets
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

# 3. Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 4. Initialize model and trainer
num_features = X_train.shape[2]  # 40 for MFCC
num_classes = len(np.unique(y_train))

model = CNN1D(
    input_size=num_features,
    num_classes=num_classes,
    num_filters=32,
    dropout_rate=0.3,
    kernel_size=3
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = CNN1DTrainer(model, device=device, learning_rate=0.001)

# 5. Train
history = trainer.train(train_loader, val_loader, epochs=50, patience=10)

# 6. Evaluate
results = trainer.evaluate(test_loader, num_classes)

# 7. Visualize
trainer.plot_learning_curves()
trainer.plot_confusion_matrix(results['confusion_matrix'])

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1-Score: {results['f1_score']:.4f}")
print(results['report'])
"""
