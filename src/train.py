import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from model_core import Two_Stream_Net
from data_loader import get_data_loaders
from metrics import calculate_metrics, get_predictions

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = Two_Stream_Net().to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/train_total, 
                            'acc': 100.*train_correct/train_total})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_labels = get_predictions(model, val_loader, device)
        val_metrics = calculate_metrics(val_labels, val_preds)
        
        print(f'\nValidation metrics:')
        for metric, value in val_metrics.items():
            print(f'{metric}: {value:.4f}')
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
        
        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
    
    # Test phase
    print('\nTesting best model...')
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_preds, test_labels = get_predictions(model, test_loader, device)
    test_metrics = calculate_metrics(test_labels, test_preds)
    
    print('\nTest metrics:')
    for metric, value in test_metrics.items():
        print(f'{metric}: {value:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Train and test face forgery detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing train/val/test subdirectories')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                      help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5,
                      help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)

if __name__ == '__main__':
    main() 