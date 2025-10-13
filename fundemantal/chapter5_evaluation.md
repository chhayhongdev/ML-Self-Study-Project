# Chapter 5: Model Evaluation and Testing

## 5.1 Evaluation Metrics

### Accuracy
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision
$$Precision = \frac{TP}{TP + FP}$$

### Recall (Sensitivity)
$$Recall = \frac{TP}{TP + FN}$$

### F1-Score
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

### Implementation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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
```

## 5.2 Confusion Matrix

### What is a Confusion Matrix?
A confusion matrix shows the predicted vs actual classifications.

### Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
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
    plt.show()

# Usage
plot_confusion_matrix(y_true, y_pred, classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
plot_confusion_matrix(y_true, y_pred, classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=True)
```

## 5.3 ROC Curves and AUC

### ROC Curve
- Plots True Positive Rate vs False Positive Rate
- Shows trade-off between sensitivity and specificity

### AUC (Area Under Curve)
- Measures overall performance
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier

### Implementation

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def plot_roc_curves(y_true, y_scores, classes):
    """Plot ROC curves for multi-class classification"""
    # Binarize labels for multi-class ROC
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
    plt.figure(figsize=(10, 8))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curves for each class
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

# Usage (y_scores should be probabilities from softmax)
plot_roc_curves(y_true, y_scores, classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
```

## 5.4 Precision-Recall Curves

### When to Use
- Better than ROC when classes are imbalanced
- Shows trade-off between precision and recall

### Implementation

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curves(y_true, y_scores, classes):
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

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, linewidth=2,
                 label=f'{classes[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Multi-class Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.show()
```

## 5.5 Cross-Validation

### K-Fold Cross-Validation
- Split data into K folds
- Train on K-1 folds, validate on remaining fold
- Repeat K times, average results

### Implementation

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_model(model_class, dataset, k=5, **model_kwargs):
    """Perform K-fold cross-validation"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k}")

        # Create data subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Initialize model
        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for a few epochs
        for epoch in range(5):
            train_epoch(model, train_loader, criterion, optimizer)

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)

        print(f"  Fold {fold+1} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    # Summary
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)

    print(f"\nCross-validation Results:")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")

    return {
        'accuracies': fold_accuracies,
        'losses': fold_losses,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }
```

## 5.6 Model Interpretability

### Grad-CAM (Gradient-weighted Class Activation Mapping)

```python
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Backward pass for target class
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

# Usage
target_layer = model.conv2  # Last convolutional layer
grad_cam = GradCAM(model, target_layer)

# Generate CAM for an image
input_image = test_dataset[0][0].unsqueeze(0).to(device)
target_class = 0  # Class index
cam = grad_cam.generate_cam(input_image, target_class)

# Visualize
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.imshow(input_image.squeeze().cpu().permute(1, 2, 0), alpha=0.5)
plt.show()
```

## 5.7 Complete Evaluation Pipeline

```python
def comprehensive_evaluation(model, test_loader, device, classes):
    """Complete model evaluation pipeline"""
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

    # 1. Basic metrics
    print("=== BASIC METRICS ===")
    metrics = calculate_metrics(all_targets, all_preds, classes)

    # 2. Confusion matrix
    print("\n=== CONFUSION MATRIX ===")
    plot_confusion_matrix(all_targets, all_preds, classes)
    plot_confusion_matrix(all_targets, all_preds, classes, normalize=True)

    # 3. ROC curves
    print("\n=== ROC CURVES ===")
    plot_roc_curves(all_targets, all_probs, classes)

    # 4. Precision-Recall curves
    print("\n=== PRECISION-RECALL CURVES ===")
    plot_precision_recall_curves(all_targets, all_probs, classes)

    return {
        'metrics': metrics,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

# Run comprehensive evaluation
results = comprehensive_evaluation(model, test_loader, device, classes)
```

---

**Next:** Chapter 6 - Model Deployment and Inference (Saving/Loading, Optimization, Production Deployment)!