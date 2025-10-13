import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import time

print("Chapter 5: Model Evaluation and Testing")
print("=" * 60)

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load a pre-trained model (or train a quick one)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
classes = test_dataset.classes

# Try to load pre-trained model, otherwise train a quick one
model = SimpleCNN().to(device)
try:
    model.load_state_dict(torch.load('simple_cnn_cifar10.pth', map_location=device))
    print("Loaded pre-trained model")
except:
    print("Training a quick model for demonstration...")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Quick training (just 2 epochs for demo)
    for epoch in range(2):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")

# 5.1 Basic Evaluation Metrics
print("\n5.1 Basic Evaluation Metrics")

def calculate_metrics(y_true, y_pred, classes):
    """Calculate comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Get predictions
model.eval()
all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

# Calculate metrics
metrics = calculate_metrics(all_targets, all_preds, classes)

# 5.2 Confusion Matrix
print("\n5.2 Confusion Matrix")

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=classes, yticklabels=classes,
                cmap='Blues')

    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(all_targets, all_preds, classes, save_path='confusion_matrix.png')
plot_confusion_matrix(all_targets, all_preds, classes, normalize=True, save_path='confusion_matrix_normalized.png')

# 5.3 ROC Curves
print("\n5.3 ROC Curves")

def plot_roc_curves(y_true, y_scores, classes, save_path=None):
    """Plot ROC curves for multi-class classification"""
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    plt.figure(figsize=(12, 8))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curves for each class (show top 5 for clarity)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(min(5, n_classes)):  # Show first 5 classes
        plt.plot(fpr[i], tpr[i], color=colors[i], linewidth=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Top 5 Classes)')
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return roc_auc

roc_aucs = plot_roc_curves(all_targets, all_probs, classes, save_path='roc_curves.png')

# 5.4 Precision-Recall Curves
print("\n5.4 Precision-Recall Curves")

def plot_precision_recall_curves(y_true, y_scores, classes, save_path=None):
    """Plot Precision-Recall curves for multi-class classification"""
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)

    # Compute precision-recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])

    # Plot (show top 5 classes)
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i in range(min(5, n_classes)):
        plt.plot(recall[i], precision[i], color=colors[i], linewidth=2,
                 label=f'{classes[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves (Top 5 Classes)')
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return average_precision

pr_aucs = plot_precision_recall_curves(all_targets, all_probs, classes, save_path='precision_recall_curves.png')

# 5.5 Model Performance Summary
print("\n5.5 Model Performance Summary")
print("-" * 40)

# Overall metrics
print(f"Overall Accuracy: {metrics['accuracy']:.2f}")
print(f"Weighted F1-Score: {metrics['f1']:.2f}")

# Best and worst performing classes
class_accuracies = []
for i in range(len(classes)):
    class_mask = (all_targets == i)
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(all_targets[class_mask], all_preds[class_mask])
        class_accuracies.append((classes[i], class_acc))

class_accuracies.sort(key=lambda x: x[1], reverse=True)

print(f"\nBest performing class: {class_accuracies[0][0]} ({class_accuracies[0][1]:.2f})")
print(f"Worst performing class: {class_accuracies[-1][0]} ({class_accuracies[-1][1]:.2f})")

# ROC AUC summary
micro_roc_auc = np.mean(list(roc_aucs.values())[:-1])  # Exclude micro average
print(f"\nMean ROC AUC: {micro_roc_auc:.3f}")

# Average Precision summary
mean_ap = np.mean(list(pr_aucs.values()))
print(f"Mean Average Precision: {mean_ap:.3f}")

# 5.6 Error Analysis
print("\n5.6 Error Analysis")

# Find most confused pairs
cm = confusion_matrix(all_targets, all_preds)
np.fill_diagonal(cm, 0)  # Remove correct predictions

max_confusion_idx = np.unravel_index(np.argmax(cm), cm.shape)
most_confused_class1 = classes[max_confusion_idx[0]]
most_confused_class2 = classes[max_confusion_idx[1]]
most_confusions = cm[max_confusion_idx]

print(f"Most confused pair: {most_confused_class1} ↔ {most_confused_class2} ({most_confusions} times)")

# Classes with highest error rates
error_rates = []
for i in range(len(classes)):
    total_samples = np.sum(all_targets == i)
    errors = np.sum((all_targets == i) & (all_preds != i))
    error_rate = errors / total_samples if total_samples > 0 else 0
    error_rates.append((classes[i], error_rate))

error_rates.sort(key=lambda x: x[1], reverse=True)
print(f"\nClass with highest error rate: {error_rates[0][0]} ({error_rates[0][1]:.2f})")

print("\n" + "="*60)
print("Chapter 5 Summary:")
print("- Comprehensive evaluation metrics calculated")
print("- Confusion matrices generated")
print("- ROC and Precision-Recall curves plotted")
print("- Per-class performance analyzed")
print("- Error patterns identified")
print("\nSaved plots: confusion_matrix.png, roc_curves.png, precision_recall_curves.png")
print("\nNext: Chapter 6 - Model Deployment and Inference!")