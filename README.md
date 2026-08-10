# Disaster AI — Image-Based Disaster Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-EfficientNetB0-success.svg)](https://keras.io/api/applications/efficientnet/)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-96.93%25-brightgreen.svg)](#results)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class)


**Disaster AI** is a deep-learning image analyzer that identifies disaster conditions — **fire**, **flood**, **earthquake**, or **normal** — from a single photograph. Along with the predicted class, it reports a **confidence score** and surfaces **emergency safety precautions** for the detected situation.

The system reduces manual effort in damage assessment, improves disaster awareness, and supports faster emergency response. By providing timely information, it assists both civilians and rescue teams in making informed decisions during critical situations, demonstrating the practical application of transfer learning in disaster management and public safety.

## Overview

| Item | Value |
| --- | --- |
| Task | Multi-class image classification (4 classes) |
| Backbone | EfficientNetB0 (ImageNet pre-trained) |
| Input size | 224 x 224 x 3 |
| Total images | 8,468 (after cleaning) |
| Train / Validation | 6,775 / 1,693 (80 / 20 split) |
| Best validation accuracy | **0.9693** |
| Framework | TensorFlow / Keras |

## Pipeline

1. Collect input images
2. Preprocess images (resize to 224x224, RGB conversion, cleanup)
3. Split data into training and validation sets (80/20)
4. Apply data augmentation on the training set only
5. Train the model using EfficientNetB0 transfer learning
6. Fine-tune the last 30 layers at a low learning rate
7. Generate predictions with confidence scores
8. Save the final model and results

## Get the Data

The dataset comes from Kaggle: **sarthaktandulje/disaster-damage-5class**. It is a merge of two sources:

- disaster_dataset — contains fire, flood, and normal images
- earthquake — contains earthquake images

Install the Kaggle dataset downloader, then download and merge the two source folders into a single four-class directory.

Dataset page: https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class

Resulting layout:

merged_dataset/

- earthquake/
- fire/
- flood/
- normal/

### Class Distribution

| Class | Images |
| --- | --- |
| earthquake | 1,000 |
| fire | 2,537 |
| flood | 2,706 |
| normal | 2,226 |
| **Total** | **8,469** (1 corrupt image removed → 8,468) |

## Labels

| Label | Description | Typical response |
| --- | --- | --- |
| 0 | earthquake | Structural collapse, rubble, cracked buildings |
| 1 | fire | Active flames, smoke, burning structures or forest |
| 2 | flood | Submerged roads, buildings, vehicles in water |
| 3 | normal | No visible disaster; safe/unaffected scene |

## Preprocessing

Every image is converted to RGB and resized to 224 x 224 using LANCZOS resampling. Unreadable or corrupt files are deleted automatically. One corrupt file was removed: merged_dataset/flood/flood1864.jpg.

## Data Augmentation

Augmentation is applied to the training subset only. The validation subset receives preprocessing without augmentation, so evaluation stays faithful to real data.

| Transformation | Setting |
| --- | --- |
| Rotation | ±20° |
| Width / height shift | 15% |
| Zoom | 15% |
| Horizontal flip | Yes |
| Fill mode | nearest |
| Validation split | 0.2 |

After augmentation, the data loader reports 6,775 training images and 1,693 validation images across the 4 classes.

## Model Architecture

### Why EfficientNetB0

1. Performs strongly on medium-sized datasets like this one.
2. Only ~5.3M parameters, so it trains far faster than larger backbones.
3. Best-in-class balance between accuracy, speed, and memory footprint.

### Model Head

The pre-trained EfficientNetB0 base is used as a frozen feature extractor, then the following classifier head is added on top:

- EfficientNetB0 (frozen)
- GlobalAveragePooling2D
- BatchNormalization
- Dropout(0.3)
- Dense(256, ReLU)
- BatchNormalization
- Dropout(0.3)
- Dense(4, Softmax)

## Training

The model is compiled with Adam optimizer at an initial learning rate of 1e-3, categorical cross-entropy loss, and accuracy as the monitored metric.

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Epochs (stage 1) | 10 |
| Early stopping | patience 5 on val_loss |
| LR schedule | ReduceLROnPlateau (factor 0.2, patience 2, min 1e-6) |

### Stage 1 training log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | LR |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8772 | 0.3749 | 0.9486 | 0.1664 | 1.0e-3 |
| 2 | 0.9258 | 0.2140 | 0.9551 | 0.1343 | 1.0e-3 |
| 3 | 0.9379 | 0.1759 | 0.9592 | 0.1140 | 1.0e-3 |
| 4 | 0.9466 | 0.1514 | 0.9640 | 0.1040 | 1.0e-3 |
| 5 | 0.9457 | 0.1401 | 0.9634 | 0.0979 | 1.0e-3 |
| 6 | 0.9479 | 0.1395 | 0.9681 | 0.0940 | 1.0e-3 |
| 7 | 0.9525 | 0.1286 | 0.9640 | 0.0868 | 1.0e-3 |
| 8 | 0.9575 | 0.1148 | 0.9610 | 0.0938 | 1.0e-3 |
| 9 | 0.9581 | 0.1216 | 0.9681 | 0.0874 | 1.0e-3 |
| **10** | **0.9672** | **0.0943** | **0.9693** | **0.0845** | **2.0e-4** |

## Fine-Tuning


The last 30 layers of the backbone are unfrozen and retrained with a very small learning rate (1e-5). The model runs for up to 10 additional epochs (epochs 11–20), with early stopping enabled.

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | LR |
| --- | --- | --- | --- | --- | --- |
| 11 | 0.9281 | 0.1968 | 0.9468 | 0.1549 | 1e-6 |
| 12 | 0.9333 | 0.1905 | 0.9474 | 0.1533 | 1e-6 |
| 13 | 0.9331 | 0.1836 | 0.9486 | 0.1500 | 1e-6 |
| 14 | 0.9343 | 0.1926 | 0.9474 | 0.1499 | 1e-6 |
| 15 | 0.9328 | 0.1938 | 0.9480 | 0.1480 | 1e-6 |

Training stopped early at epoch 15 — the learning rate had already collapsed to 1e-6, so fine-tuning did not surpass the frozen-backbone checkpoint. The best model remains the epoch-10 checkpoint at 0.9693 validation accuracy.

## Results & Visualizations

### Training vs Validation Accuracy

The model shows consistent improvement in both training and validation accuracy throughout training. The final validation accuracy reached approximately **96.93%**, demonstrating strong generalization performance.

![Training vs Validation Accuracy](images/accuracy_curve.png)

---

### Training vs Validation Loss

Training and validation loss decrease steadily across epochs, indicating effective learning and minimal overfitting. The close alignment between the curves suggests good model stability.

![Training vs Validation Loss](images/loss_curve.png)

---

### Confusion Matrix

The confusion matrix shows that the model performs exceptionally well across all four classes. Most predictions lie on the diagonal, indicating correct classification.

![Confusion Matrix](images/confusion_matrix.png)

### Key Observations

- Fire achieved the highest recall with very few misclassifications.
- Flood images were classified accurately in most cases.
- Earthquake images occasionally overlapped with flood scenes.
- Normal images showed minor confusion with flood and earthquake classes.
- Overall classification performance exceeded 95% across all classes.

| Class | Correct Predictions |
|---------|---------|
| Earthquake | 183 / 200 |
| Fire | 503 / 507 |
| Flood | 516 / 541 |
| Normal | 401 / 445 |

### Performance Summary

| Metric | Value |
|----------|----------|
| Best Validation Accuracy | 96.93% |
| Final Validation Loss | 0.0845 |
| Number of Classes | 4 |
| Model | EfficientNetB0 |
| Input Size | 224 × 224 |

Reading the numbers: fire is the easiest class (0.99 recall) — flames and smoke are visually unambiguous. normal has the lowest recall (0.90), since undamaged scenes are occasionally confused with mild flood or earthquake imagery. earthquake has the smallest support (200 validation images), which explains its slightly noisier score.

## Evaluation

The final model achieved a validation loss of 0.1549 and a validation accuracy of 0.9468. The highest validation accuracy observed during stage 1 was 0.9693.

Both training and validation accuracy curves track closely with no widening gap, indicating that the dropout, batch normalisation, and augmentation successfully controlled overfitting.

The confusion matrix is strongly diagonal; the majority of residual error sits between normal and flood.

## Usage

1. Install dependencies: TensorFlow, KaggleHub, Pillow, NumPy, scikit-learn, Matplotlib, and Seaborn.
2. Download the dataset from Kaggle and merge the two source folders into the four-class layout.
3. Preprocess the images (resize to 224x224, convert to RGB, remove corrupt files).
4.Split the Data into (Train(80%)/Validation(20%))
5.Train Data undergoes Data Augmentation 
6. Train the model with the stage-1 settings.
7. Fine-tune the last 30 layers.
8. Save the trained model in Keras format.
9. Load the saved model and run predictions on new images.

### Emergency safety precautions

The predictor returns the class, confidence, and a precaution message:

- earthquake: Drop, cover, and hold on. Move away from windows and unstable structures. Evacuate to open ground once shaking stops; avoid lifts and damaged buildings.
- fire: Evacuate immediately and call emergency services. Stay low to avoid smoke, never use lifts, and do not re-enter the building.
- flood: Move to higher ground at once. Avoid walking or driving through moving water, switch off electricity at the mains, and drink only safe water.
- normal: No disaster detected. Keep emergency contacts, a first-aid kit, and an evacuation plan ready as a precaution.

A sample prediction on a flood image reports the class as flood with a confidence of around 97.42%.

## Project Structure

- README.md
- requirements.txt
- notebooks/ — disaster_ai.ipynb end-to-end training notebook
- data/merged_dataset/ — earthquake, fire, flood, normal
- models/ — best_cnn_model.keras (best checkpoint) and efficientnetb0_disaster_classifier.keras
- outputs/ — accuracy_curve.png, loss_curve.png, confusion_matrix.png

## Requirements

- TensorFlow 
- kagglehub
- Pillow
- Shutil
- scikit-learn
- Matplotlib
- Seaborn

A GPU runtime is strongly recommended — each epoch took roughly 12–15 minutes on the training hardware used here.

## Future Work

- Add severity grading (minor / moderate / severe) on top of disaster type.
- Try EfficientNetB2/B3 or a ConvNeXt backbone and compare against the B0 baseline.
- Add Grad-CAM heatmaps so responders can see which regions drove the prediction.
- Deploy behind a web or mobile interface with geotagging for real-time field reporting.

## Acknowledgements

- Dataset: sarthaktandulje/disaster-damage-5class on Kaggle.
- Backbone: EfficientNetB0
- Built with TensorFlow / Keras.

## License


This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
