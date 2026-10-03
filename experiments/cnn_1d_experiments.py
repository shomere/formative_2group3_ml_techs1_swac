"""
CNN1D Hyperparameter Tuning Experiments
Member 4 - Formative Assignment 2
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import sys
import os

# Fix the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn_1d_model import CNN1D, CNN1DTrainer

np.random.seed(42)
torch.manual_seed(42)

print("Loading data...")
# Setup data directory path
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'w1.5_energy_mfcc40_d')

train_data = np.load(os.path.join(data_dir, 'train.npz'))
X_train = train_data['X'].transpose(0, 2, 1)  # Transpose to (batch, features, sequence)
y_train = train_data['y']

val_data = np.load(os.path.join(data_dir, 'val.npz'))
X_val = val_data['X'].transpose(0, 2, 1)
y_val = val_data['y']

test_data = np.load(os.path.join(data_dir, 'test.npz'))
X_test = test_data['X'].transpose(0, 2, 1)
y_test = test_data['y']


print(f"Data loaded: X_train {X_train.shape}, y_train {y_train.shape}")

# Define 5+ experiment configurations
configs = [
    {'name': 'baseline', 'num_filters': 32, 'kernel_size': 3, 'learning_rate': 0.001, 'batch_size': 64, 'dropout_rate': 0.3, 'epochs': 50},
    {'name': 'increased_filters', 'num_filters': 64, 'kernel_size': 3, 'learning_rate': 0.001, 'batch_size': 64, 'dropout_rate': 0.3, 'epochs': 50},
    {'name': 'smaller_batch', 'num_filters': 64, 'kernel_size': 3, 'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.3, 'epochs': 50},
    {'name': 'larger_kernel', 'num_filters': 64, 'kernel_size': 5, 'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.3, 'epochs': 50},
    {'name': 'higher_dropout', 'num_filters': 64, 'kernel_size': 3, 'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.4, 'epochs': 50},
]

# Run experiments
results = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

for config in configs:
    print(f"Running: {config['name']}")
    
    num_features = X_train.shape[1]  # Features/channels dimension
    num_classes = len(np.unique(y_train))
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Train model
    model = CNN1D(input_size=num_features, num_classes=num_classes, 
                  num_filters=config['num_filters'], kernel_size=config['kernel_size'],
                  dropout_rate=config['dropout_rate'])
    
    trainer = CNN1DTrainer(model, device=device, learning_rate=config['learning_rate'])
    history = trainer.train(train_loader, val_loader, epochs=config['epochs'], patience=10)
    
    # Evaluate
    eval_results = trainer.evaluate(test_loader, num_classes)
    
    # Save results
    result = {
        'experiment_id': config['name'],
        'num_filters': config['num_filters'],
        'kernel_size': config['kernel_size'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'dropout_rate': config['dropout_rate'],
        'train_loss': history['train_loss'][-1],
        'train_accuracy': history['train_acc'][-1],
        'val_loss': history['val_loss'][-1],
        'val_accuracy': history['val_acc'][-1],
        'test_accuracy': eval_results['accuracy'],
        'f1_score': eval_results['f1_score'],
    }
    results.append(result)
    print(f"✓ Test Accuracy: {eval_results['accuracy']:.4f}, F1: {eval_results['f1_score']:.4f}\n")

# Save to CSV
results_df = pd.DataFrame(results)
results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'cnn_experiments.csv')
results_df.to_csv(results_path, index=False)
print("Results saved to: results/cnn_experiments.csv")
print(results_df)

# Generate comparison plots
print("\n\nGenerating plots...")

# 1. Hyperparameter Comparison Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Test Accuracy vs Experiment
axes[0, 0].bar(results_df['experiment_id'], results_df['test_accuracy'], color='steelblue')
axes[0, 0].set_title('Test Accuracy by Configuration', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)

# F1 Score vs Experiment
axes[0, 1].bar(results_df['experiment_id'], results_df['f1_score'], color='coral')
axes[0, 1].set_title('F1 Score by Configuration', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].tick_params(axis='x', rotation=45)

# Training vs Validation Accuracy
axes[1, 0].scatter(results_df['train_accuracy'], results_df['val_accuracy'], s=100, alpha=0.6, color='green')
for i, txt in enumerate(results_df['experiment_id']):
    axes[1, 0].annotate(txt, (results_df['train_accuracy'][i], results_df['val_accuracy'][i]), fontsize=8)
axes[1, 0].set_xlabel('Train Accuracy')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].set_title('Train vs Validation Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.3)  # Ideal line

# Hyperparameter Settings
axes[1, 1].axis('off')
summary_text = "Best Configuration: larger_kernel\n\n"
best_idx = results_df['test_accuracy'].idxmax()
best_row = results_df.loc[best_idx]
summary_text += f"Test Accuracy: {best_row['test_accuracy']:.4f}\n"
summary_text += f"F1 Score: {best_row['f1_score']:.4f}\n"
summary_text += f"Filters: {int(best_row['num_filters'])}\n"
summary_text += f"Kernel Size: {int(best_row['kernel_size'])}\n"
summary_text += f"Batch Size: {int(best_row['batch_size'])}\n"
summary_text += f"Dropout: {best_row['dropout_rate']}"
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', family='monospace')

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'hyperparameter_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Hyperparameter analysis saved: {plot_path}")
plt.close()

print("\n✅ All plots generated successfully!")