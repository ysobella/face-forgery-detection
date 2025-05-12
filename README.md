# Face Forgery Detection

This project implements a two-stream neural network for detecting face forgeries in images. The model uses a combination of spatial and frequency domain features to identify manipulated images.

## Project Structure

```
.
├── src/
│   ├── model_core.py      # Two-stream network architecture
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│   └── metrics.py        # Evaluation metrics
├── outputs/              # Directory for saved models and results
└── README.md            # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- PIL
- tqdm
- scikit-learn

## Data Organization

The dataset should be organized in the following structure:

```
data_dir/
├── train/
│   ├── real/
│   │   └── [real images]
│   └── fake/
│       └── [fake images]
├── val/
│   ├── real/
│   │   └── [real images]
│   └── fake/
│       └── [fake images]
└── test/
    ├── real/
    │   └── [real images]
    └── fake/
        └── [fake images]
```

## Training

To train the model, use the following command:

```bash
python src/train.py --data_dir /path/to/data --output_dir outputs
```

Training parameters:
- Batch size: 8
- Number of epochs: 100
- Number of workers: 8
- Early stopping patience: 5 epochs
- Learning rate: 0.001 (default)

Additional options:
```bash
python src/train.py \
    --data_dir /path/to/data \
    --output_dir outputs \
    --batch_size 8 \
    --num_workers 8 \
    --learning_rate 0.001 \
    --epochs 100 \
    --patience 5
```

The training script will:
1. Load and preprocess the data (only normalization, no resizing)
2. Train the model with early stopping
3. Save the best model based on validation F1-score
4. Evaluate the model on the test set

## Testing

### Test a Single Image

To test a single image:

```bash
python src/test.py \
    --model_path outputs/best_model.pth \
    --single_image /path/to/image.jpg
```

### Test a Directory

To test all images in a directory:

```bash
python src/test.py \
    --model_path outputs/best_model.pth \
    --test_dir /path/to/test/directory
```

The test directory should follow the same structure as the training data:
```
test_dir/
├── real/
│   └── [real images]
└── fake/
    └── [fake images]
```

## Model Architecture

The model uses a two-stream architecture:
1. Spatial Stream: Processes the original image
2. Frequency Stream: Processes the frequency domain representation

Key features:
- No image resizing (preserves original image dimensions)
- Only normalization is applied to input images
- CrossEntropyLoss for training
- Early stopping to prevent overfitting

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Notes

- The model preserves original image dimensions and only applies normalization
- Early stopping helps prevent overfitting
- The best model is saved based on validation F1-score
- GPU is used if available, otherwise falls back to CPU

## Overview

In this paper, we find that current CNN-based detectors tend to overfit to method-specific color textures and thus fail to generalize. Observing that image noises remove color textures and expose discrepancies between authentic and tampered regions, we propose to utilize the high-frequency noises for face forgery detection.

We carefully devise three functional modules to take full advantage of the high-frequency features. 

- The first is the multi-scale high-frequency feature extraction module that extracts high-frequency noises at multiple scales and composes a novel modality. 
- The second is the residual-guided spatial attention module that guides the low-level RGB feature extractor to concentrate more on forgery traces from a new perspective. 
- The last is the cross-modality attention module that leverages the correlation between the two complementary modalities to promote feature learning for each other. 

The two-stream model is shown as follows.

![image-20210428105010020](img/pipeline.png)

## Dependency

The model is implemented with PyTorch.

Pretrained Xception weights are downloaded from [this link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth).

## Contact

Please contact - LUO Yuchen - 592mcavoy@sjtu.edu.cn

## Citation

```
@InProceedings{Luo_2021_CVPR,
    author    = {Luo, Yuchen and Zhang, Yong and Yan, Junchi and Liu, Wei},
    title     = {Generalizing Face Forgery Detection With High-Frequency Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16317-16326}
}
```

## Notice
Thank you all for using this repo! I've received several emails regarding to the implementation and reproducing issues. Here I list some tips that might be useful :).
- Training and testing are conducted following the specifications in FaceForensics++ [paper](https://arxiv.org/abs/1901.08971).
- The training datasets can be downloaded and created following the official instructions of [FaceForensics++](https://github.com/ondyari/FaceForensics).
- The cross-dataset performance is largely influenced by the training data scale and training time. I found that the GPU version, the training batchsize, and the distributing setting also have impacts on the performance. Please refer to the detailed specifications in the paper.









