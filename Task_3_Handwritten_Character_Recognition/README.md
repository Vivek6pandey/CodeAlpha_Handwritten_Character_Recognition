# CodeAlpha ML Internship — Task 3: Handwritten Character Recognition

## Objective
Identify handwritten characters/digits using image processing and deep learning.

## Dataset
MNIST handwritten digit dataset. The dataset contains 28×28 grayscale images of digits 0–9 and is downloaded automatically by Keras.

## Model
A Convolutional Neural Network (CNN) with:
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Flatten
- Dense layer
- Dropout
- Softmax output layer

## Preprocessing
Pixel values are normalized from 0–255 to 0–1 and a channel dimension is added.

## Evaluation
Accuracy, classification report and confusion matrix are generated. Training/validation accuracy is also plotted.

## Run
```bash
pip install -r requirements.txt
python handwritten_character_recognition.py
```

The script saves the trained Keras model and result images.
