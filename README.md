# Liveness Detection System

Prototype of a liveness detection system to identify spoofs in videos from camera selfies. The project includes training, evaluation, inference and adversarial attack generation to test and improve the liveness detection model.

## Table of Contents

- **[Comprehensive project report](./Report.md)**
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [License](./LICENSE)

## Installation

### Using Conda

1. Clone the repository:
    ```sh
    git clone https://github.com/andreluizbvs/liveness_system.git
    cd liveness_system
    ```

2. Create and activate the Conda environment:
    ```sh
    conda env create -f environment.yml
    conda activate liveness
    ```
    or

    ```sh
    conda create -n liveness python=3.12
    pip install -r requirements.txt
    conda install -c conda-forge libstdcxx-ng
    conda activate liveness
    ```

## Usage

1. Prepare model weights and datasets. Place datasets in the `data/` folder, and the weights in the `ckpt/` folder, both at this project's root directory

2. Run the inference scripts to see the system working on an image or a video:
    ```sh
    cd src/
    ```
    ```sh
    python liveness_inference.py 
    ```
    ```sh
    python adv_attack_inference.py
    ```

3. To fine-tune it, run the train script to train and evaluate the liveness detection model:
    ```sh
    python train.py
    ```
    It will output the accuracy, precision, recall, and F1-score of the model.


## References

A few inspiring projects with insightful content that adapted here:

- [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) (Dataset and AENet)
- [[CVPR2024 Workshop] Joint Physical-Digital Facial Attack Detection Via Simulating Spoofing Clues](https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge) (Creative data augmentation)
- https://www.kaggle.com/code/duchuy/face-anti-spoofing (Part of the dataloader for TF)
- https://www.kaggle.com/datasets/trainingdatapro/silicone-masks-biometric-attacks/data (For the silicone mask dataset)