"""
Evaluation Metrics for Face Forgery Detection

Provides utility functions to compute accuracy, precision, recall,
and F1-score, as well as to collect predictions from a model.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing all metrics
    """
    # Convert to numpy arrays if they're tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def get_predictions(model, dataloader, device):
    """
    Generate predictions and ground truth labels from a DataLoader.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): DataLoader containing dataset.
        device (torch.device): Device to perform inference on.

    Returns:
        tuple: (predictions, true_labels) as NumPy arrays.
    """
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs, _, _ = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels) 