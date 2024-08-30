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

## Design choices

- Since it is a very hard problem, I chose to use the outputs of multiple models in order to have a very well-informed live/spoof prediction. That slows down the output but, not too slow so that a person using it would be annoyed. The use of ONNX was to mitigate that waiting time, which was successful.
- Data flow is very simple and easy to follow so new datasets can be easily added.
- I thought about creating a well-organized and modularized project to enable future contributions, updates and general additions to it.

## Performance Metrics

Classical classification metrics:

- **Accuracy** (Since that datasets used here are balanced)
- **F1-Score**
- **Precision**
- **Recall**

## Technologies

Main Deep Learning frameworks and popular computer vision libraries, as well as other import tools:

- TensorFlow (for performance [and I'm more used to it for the moment])
- PyTorch (for fast prototyping, more common on research & scientific environments, as the problem needs)
- DeepFace (face bounding box extraction for dataset creation, and of course for inference on "production-like" data)
- InsightFace (for landmarks extraction for binary mask in adversarial attack generation)
- OpenCV and Pillow (for general image operations)
- Albumentations (for more rich data augmentation operations)
- ONNX & ONNX Runtime(for high performance and real-time inference)
- Other minor supporting libs
- Of course, the Python language + Python Notebooks for really fast prototyping
- Github Copilot for productivity boost
- Nvidia GPU and CUDA for faster training, evaluation and inference times


## File Details

### Root Directory
- **environment.yml**: Conda environment configuration file.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **setup.py**: Script for setting up the project package.

### `src/`
- **`adv_attack_inference.py`**: Script for running adversarial attack inference.
- **`evaluate.py`**: Script for evaluating the all models.
- **`liveness_inference.py`**: Script for running liveness detection inference on an image or video file.
- **`train.py`**: Script for training the liveness detection models (FaceDepth or SiliconeMask). Futurely, it is intended that AENet also could be fine-tuned.

### `src/dataloader/`
- **`__init__.py`**: Initializes the `dataloader` module.
- **`adversarial_attack_data_aug.py`**: Contains multiple data augmentation techniques for adversarial attacks.
- **`celeba_spoof.py`**: Handles the CelebA Spoof dataset.
- **`dataset.py`**: General dataset handling utilities.


### `src/models/`

- **FaceDepth**: This model is responsible for depth-based liveness detection. Its pretrained backbone estimates a depth map (monocular-depth-estimation) and then this is passed to binary classification head. The idea here is to identify spoofs such as photos of screen images (faces) and photos of printed faces. The model is trained is the CelebA-Spoof dataset.

- **SiliconeMask**: This model focuses on classifying if a face corresponds to a real face (live) of a face of a person wearing a silicone mask (spoof), which are commonly used in such attacks. A regular backbone (ResNet50) is used to extract a feature embedding, which is then passed to a binary classification head.

- **AENet**: AENet stands for Auxiliary Information Embedding Network. This model was published together with the CelebA-Spoof dataset in ECCV 2020 (one the most important events of Computer Vision in the world). AENet is trained using a combination of clean and physically-cracfted adversarial examples (CelebA-Spoof dataset) to improve its robustness against adversarial attacks.

- **AdversarialAttack**: This model is responsible for generating adversarial examples using various attack techniques. These include the introduction of moire effect, facial artifact, color distortion and gradient noise.

- **Other models**: In addition to the mentioned models (FaceDepth, SiliconeMask, AENet), there may be additional models in the `src/models/` directory that are specific to different aspects of liveness detection. These models could include face swapping detection or detection of other types of spoofing attacks. However, a representative dataset with this kind of data could be enough to finetune a model like AENet.


### `src/tools/`
- **`liveness_predict.ipynb`**: A jupyter notebook to easily test the generalist Liveness model input images. Predictions close to 0 mean "live" while predictions close to 1 mean "spoof"; 
- **`adversarial_attack_manipulation.ipynb`**: A jupyter notebook depicting the generation of an adversarial attack image (spoof), by applying different techniques of data augmentation, such as moire effect, facial artifact, color distortion, gradient noise some only on the subject face. This is done by extract the binarized segmentation mask of the subject's face through its detected facial landmarks.
- **`liveness_output_analysis`**: A jupyter notebook showing an analysis on every output provided by AENet, such as depth map and reflection map predictions, and how they related to some characteristics of lives and spoofs.
- **`gen_image_data.py`**: Script for generating image data from video frames and face data from whole-image data.
- **`convert_keras_model.py`** and **`convert_torch_model.py`**: Scripts to optimize models by converting them to the ONNX format.

### `src/utils/`
- `security.py`: A future implementation of a function that identifies which types of adversarial attacks the model is most vulnerable by providing it annotations of each type of attack in a given dataset.


## Conclusion

The project attempts to tackle the liveness and adversarial attack problems through multiple angles. Its structure was thought and organized to facilitate future additions and upgrades, such as new models, new datasets, and model optimizations for real-time perfomance.