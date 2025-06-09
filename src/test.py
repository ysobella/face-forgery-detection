"""
Test Script for Face Forgery Detection Model

This script allows testing a trained Two-Stream model on either:
- A single image (for real/fake classification)
- A directory of test images (structured into 'real' and 'fake' folders)

Usage:
    Single Image:
        python test.py --model_path path/to/model.pth --single_image path/to/image.jpg

    Directory:
        python test.py --model_path path/to/model.pth --test_dir path/to/test_dir
"""

import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from model_core import Two_Stream_Net
from metrics import calculate_metrics

def load_image(image_path):
    """
    Load and preprocess a single image.

    Args:
        image_path (str): Path to the image.

    Returns:
        Tensor: Preprocessed image tensor with batch dimension.
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_single_image(model, image_path, device):
    """
    Predict whether a single image is real or fake.

    Args:
        model (nn.Module): Trained model.
        image_path (str): Path to the image.
        device (torch.device): Device to run inference on.

    Returns:
        dict: Prediction result containing class, confidence, and attention map.
    """
    
    model.eval()
    image = load_image(image_path).to(device)
    
    with torch.no_grad():
        output, _, attention_map = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.softmax(output, dim=1)
        
    return {
        'prediction': 'fake' if predicted.item() == 1 else 'real',
        'confidence': probability[0][predicted].item(),
        'attention_map': attention_map
    }

def test_directory(model, test_dir, device):
    """
    Test all images in a directory (expects 'real' and 'fake' subfolders).

    Args:
        model (nn.Module): Trained model.
        test_dir (str): Path to test directory.
        device (torch.device): Device to run inference on.

    Returns:
        dict: Classification metrics for the test set.
    """
    
    real_dir = os.path.join(test_dir, 'real')
    fake_dir = os.path.join(test_dir, 'fake')
    
    all_preds = []
    all_labels = []
    
    # Test real images
    for img_name in os.listdir(real_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(real_dir, img_name)
            result = predict_single_image(model, img_path, device)
            all_preds.append(1 if result['prediction'] == 'fake' else 0)
            all_labels.append(0)  # 0 for real
    
    # Test fake images
    for img_name in os.listdir(fake_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(fake_dir, img_name)
            result = predict_single_image(model, img_path, device)
            all_preds.append(1 if result['prediction'] == 'fake' else 0)
            all_labels.append(1)  # 1 for fake
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    return metrics

def main():
    """
    Main entry point. Loads model, performs testing, and prints results.
    """
    
    parser = argparse.ArgumentParser(description='Test face forgery detection model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Directory containing test images (with real/fake subdirectories)')
    parser.add_argument('--single_image', type=str,
                      help='Path to a single image to test (optional)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = Two_Stream_Net().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    if args.single_image:
        # Test single image
        result = predict_single_image(model, args.single_image, device)
        print(f'\nPrediction for {args.single_image}:')
        print(f'Class: {result["prediction"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
    else:
        # Test directory
        metrics = test_directory(model, args.test_dir, device)
        print('\nTest metrics:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main() 