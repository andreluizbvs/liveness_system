# Report

## Project Overview

This project implements a liveness detection system using machine learning and computer vision to distinguish between live individuals and non-live entities (spoofs). The project includes training, evaluation, inference, and adversarial attack generation to test and improve the liveness detection model.

## Methodologies

### Liveness Detection Algorithm
- **Model Selection**: All are CNN-based. Facial depth estimation, embedding extraction (ResNet) for mask detection, and AENet (for reflection estimation and a binary general classification of live/spoof images)
- **Feature Engineering**: Image preprocessing, traditional data augmentation and Face-anti spoofing dataset augmentation 
- **Training Process**: Embedding extraction model was trained on a dataset made of video frames containing real people using or not 3d silicone masks. AENet may be finetuned with adversarial attack samples generated the FAS data augmentation.

### Adversarial Machine Learning
- **Adversarial Examples**: Face-anti spoofing dataset augmentation: moire effect, facial artifact, color distortion, gradient noise.
- **Robustness Enhancement**: Model finetuning with adversarial attack examples to improve model robustness

### Security Measures
- **Vulnerabilities**: To be identified.
- **Mitigation**: Data augmentation (introduction of moire effect, facial artifact, color distortion, gradient noise)

## System Architecture

![Architecture]()

## Data Flow Diagram

![Data Flow](data_flow_diagram.png)

## Challenges and Solutions

- **Challenge**: A general solution to detect all main types of face spoofs
- **Solution**: Decision made based on the output of multiple models trainined in different datasets and contexts. Almost as an emsemble.

## Performance Metrics

Classical classification metrics:

- **Accuracy** (Since that datasets used here are balanced)
- **F1-Score**
- **Precision**
- **Recall**

## File Details

### Root Directory
- **environment.yml**: Conda environment configuration file.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **setup.py**: Script for setting up the project package.

### `src/`
- **`adv_attack_inference.py`**: Script for running adversarial attack inference.
- **`evaluate.py`**: Script for evaluating the all models.
- **`liveness_inference.py`**: Script for running liveness detection inference on an image or video file.
- **`train.py`**: Script for training the liveness detection models (FaceDepth, SiliconeMask, AENet).

### `src/dataloader/`
- **`__init__.py`**: Initializes the `dataloader` module.
- **`adversarial_attack_data_aug.py`**: Contains multiple data augmentation techniques for adversarial attacks.
- **`celeba_spoof.py`**: Handles the CelebA Spoof dataset.
- **`dataset.py`**: General dataset handling utilities.



### `src/models/`

- **FaceDepth**: This model is responsible for depth-based liveness detection. It uses depth information extracted from facial images to distinguish between live individuals and spoofs. The model is trained on a dataset of depth maps generated from RGB-D images.

- **SiliconeMask**: This model focuses on detecting silicone masks, which are commonly used in spoofing attacks. It utilizes a combination of image processing techniques and machine learning algorithms to identify the presence of silicone masks on facial images.

- **AENet**: AENet stands for Auxiliary Information Embedding Network. This model is designed to detect adversarial examples, which are carefully crafted inputs that can fool machine learning models. AENet is trained using a combination of clean and physically-cracfted adversarial examples (CelebA-Spoof dataset) to improve its robustness against adversarial attacks.

- **AdversarialAttack**: This model is responsible for generating adversarial examples using various attack techniques. These include the introduction of moire effect, facial artifact, color distortion and gradient noise.


- **Other models**: In addition to the mentioned models (FaceDepth, SiliconeMask, AENet), there may be additional models in the `src/models/` directory that are specific to different aspects of liveness detection. These models could include face swapping detection or detection of other types of spoofing attacks. Unfortunately, the details of these additional models are not available in the provided markdown.


### `src/tools/`
- `gen_image_data.py`**: Script for generating image data from video frames and face data from whole-image data.

### `src/utils/`
- `security.py`: A future implementation of a function that identifies which types of adversarial attacks the model is most vulnerable by providing it annotations of each type of attack in a given dataset.


## Conclusion

The project attempts to tackle the liveness and adversarial attack problems through multiple angles. Its structure was thought and organized to facilitate future additions and upgrades, such as new models, new datasets, and model optimizations for real-time perfomance.