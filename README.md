# Brain Tumor Detection with CNN

This project implements a Convolutional Neural Network (CNN) for brain tumor detection and classification using medical image data. The model is designed to classify images into four different types of brain tumors.

## Features

- **Multi-class Classification:** The model processes images belonging to four different tumor types and predicts the correct class for each input image.
- **End-to-End Pipeline:** Includes data loading, preprocessing (resizing, normalization), model building, training, evaluation, and visualization of training history.
- **Performance Evaluation:** After training, the model's accuracy and loss are measured on a separate validation set.

## Dataset

- The dataset should be organized into subfolders for each tumor type, both for training and testing.
- **Note:** The dataset used in this project is not very large or diverse. Using a more comprehensive and higher-quality dataset would significantly improve the model's performance and generalization.

You can acces dataset from here:
"https://www.kaggle.com/datasets/orvile/brain-tumor-dataset"

## Usage

1. Place your dataset in the appropriate folder structure:
    ```
    Brain Tumor data/
        Brain Tumor data/
            Training/
                TumorType1/
                TumorType2/
                TumorType3/
                TumorType4/
            Testing/
                TumorType1/
                TumorType2/
                TumorType3/
                TumorType4/
    ```
2. Run the `TumorDetection.py` script to train and evaluate the model.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install dependencies with:
```bash
pip install tensorflow numpy matplotlib
```

## Results

- The script prints the training and validation accuracy and loss after each epoch.
- At the end, it reports the final test accuracy and loss.
- Training history is visualized with plots for both accuracy and loss.

## Limitations

- **Dataset Quality:** The current dataset is limited in size and diversity. For real-world applications and higher accuracy, it is highly recommended to use a larger and more representative dataset.

---

*This project is for educational purposes and should not be used for clinical decision-making.*
