# Liveness Detection System

Prototype of a liveness detection system to identify spoofs in videos from camera selfies. The project includes training, evaluation, inference and adversarial attack testing of the liveness detection model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](./LICENSE)

## Installation

### Using Conda

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/liveness_detection_system.git
    cd liveness_detection_system
    ```

2. Create and activate the Conda environment:
    ```sh
    conda env create -f environment.yml
    conda activate liveness_detection
    ```

### Using Virtualenv

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/liveness_detection_system.git
    cd liveness_detection_system
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place it in the `data/frames` directory.

2. Run the main script to train and evaluate the liveness detection model:
    ```sh
    python src/main.py
    ```

3. The script will output the accuracy, precision, recall, and F1-score of the model.


## References

A few projects with insightful content that were used here:

- [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) (Dataset and AENet)
- [[CVPR2024 Workshop] Joint Physical-Digital Facial Attack Detection Via Simulating Spoofing Clues](https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge) (Creative data augmentation)
- https://www.kaggle.com/code/duchuy/face-anti-spoofing (Part of the dataloader for TF)