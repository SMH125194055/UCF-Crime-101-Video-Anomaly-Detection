# Anomaly Detection in Surveillance Videos using CNN-LSTM

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TPU](https://img.shields.io/badge/Google%20Cloud-TPU-orange?style=for-the-badge&logo=google-cloud)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

A deep learning approach for anomaly detection in surveillance videos using CNN-LSTM architecture with triplet loss, trained on TPUs.

## 📌 Overview

This project implements a hybrid CNN-LSTM model for classifying anomalous activities in surveillance videos from the UCF-Crime dataset. The model combines:
- **ResNet18** for spatial feature extraction
- **Bi-directional LSTM** with attention mechanism for temporal modeling
- **Triplet loss** for improved feature embedding
- **TPU acceleration** for faster training

## 🏆 Key Features

- **Hybrid Architecture**: Combines CNN and LSTM for spatio-temporal feature learning
- **Attention Mechanism**: Focuses on relevant frames in video sequences
- **Triplet Loss**: Improves discriminative power of embeddings
- **TPU Optimization**: Leverages Google Cloud TPUs for accelerated training
- **Visualization Tools**: Includes embedding visualization and prediction samples

## 📚 Dataset

We use the **UCF-Crime** dataset containing 1900 long untrimmed surveillance videos across 14 classes:

| Class | # Videos | Class | # Videos |
|-------|----------|-------|----------|
| Normal | 800 | Abuse | 50 |
| Arrest | 50 | Arson | 50 |
| Assault | 50 | Burglary | 50 |
| Explosion | 50 | Fighting | 50 |
| RoadAccidents | 50 | Robbery | 50 |
| Shooting | 50 | Shoplifting | 50 |
| Stealing | 50 | Vandalism | 50 |

## 🛠️ Implementation

### Model Architecture

```python
class Enhanced_CNN_LSTM(nn.Module):
    def __init__(self, num_classes=14, embedding_dim=128, use_triplet=False):
        super().__init__()
        # Pretrained ResNet18 backbone
        self.cnn = torchvision.models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        
        # Final layers
        self.fc = nn.Linear(512, embedding_dim) if use_triplet else nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes))
