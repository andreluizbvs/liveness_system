# Liveness Detection System

## Overview

This project implements a liveness detection system using machine learning and computer vision to distinguish between live individuals and non-live entities (spoofs).
This project also proposes the generation of adversarial attacks to:
    - Attempt to bypass the system
    - Strengthen the model's anti-spoof capabilities 

## Methodologies

### Liveness Detection Algorithm
- **Model Selection**: Convolutional Neural Network (CNN)
- **Feature Engineering**: Image preprocessing, data augmentation
- **Training Process**: Trained on a dataset of live and non-live images

### Adversarial Machine Learning
- **Adversarial Examples**: 
- **Robustness Enhancement**: Adversarial training

### Security Measures
- **Vulnerabilities**: Basic spoofing attacks (photo/video of a screen, printed face photo/video, paper masks), silicone masks, GenAI image manipulations (e.g., face swaping)
- **Mitigation**: Data augmentation (introduction of moire effect, facial artifact, color distortion, gradient noise)

## System Architecture

![Architecture](architecture.png)

## Data Flow Diagram

![Data Flow](data_flow_diagram.png)

## Challenges and Solutions

- **Challenge**: Adversarial robustness
- **Solution**: 

## Performance Metrics

- **Accuracy**:
- **F1-Score**:
- **Precision**: 
- **Recall**: 