# Oscilloscope Waveform Classifier

A computer vision project that uses a **Convolutional Neural Network (CNN)** to classify waveform images captured from an oscilloscope.

The model identifies four common waveform types:

- Sine Wave
- Square Wave
- Triangle Wave
- Sawtooth Wave

## Project Overview

The goal of this project is to develop a machine learning system capable of automatically recognizing waveform types from oscilloscope images.

Instead of analyzing raw voltage data directly, the system processes the **visual waveform displayed on an oscilloscope**, allowing waveform classification from screenshots or photographs of real test equipment.

The project combines concepts from:

- Electrical Engineering
- Digital Signal Analysis
- Computer Vision
- Machine Learning
- Image Processing

## How It Works

The classification pipeline follows:

```text
Oscilloscope Image
        ↓
Image Cropping
        ↓
Resize / Preprocessing
        ↓
Convolutional Neural Network
        ↓
Waveform Classification
        ↓
Sine / Square / Triangle / Sawtooth
```

Images are cropped to isolate the waveform display and resized to the dimensions expected by the neural network.

The trained CNN then extracts visual features from the waveform and predicts the most likely waveform class.

## Model

The classifier is built using **TensorFlow / Keras**.

Current image dimensions:

```python
IMG_HEIGHT = 300
IMG_WIDTH = 600
```

The dataset uses an **80/20 training and validation split**:

```python
validation_split = 0.2
seed = 123
```

The trained model is stored as:

```text
waveform_classifier.keras
```

Class labels are stored in:

```text
class_names.txt
```

## Image Preprocessing

Oscilloscope images contain additional information such as menus, measurements, grid markings, and controls that are not directly relevant to waveform classification.

A fixed crop is applied to focus the model on the waveform display.

Current cropping region:

```text
Top:     6%
Bottom:  82%
Left:    2%
Right:   98%
```

The cropped image is then resized before being passed to the CNN.

## Dataset

The model is trained using images representing four waveform classes:

```text
waveform_dataset/
│
├── sine/
├── square/
├── triangle/
└── sawtooth/
```

A separate testing dataset is used to evaluate the model on images not included during training:

```text
waveform_testingset/
```

## Example Prediction

The model outputs the predicted waveform along with its confidence score.

Example:

```text
Predicted Waveform: Sine
Confidence: 99.83%
```

## Current Challenges

One of the primary challenges is distinguishing between waveforms with similar visual characteristics. During testing, some **square waves were incorrectly classified as triangle waves**.

Additional challenges include:

- Different oscilloscope screen layouts
- Image rotation and perspective distortion
- Screen glare and reflections
- Different waveform frequencies and amplitudes
- Signal noise and distortion
- Different voltage and time scales
- Photographs taken from different angles

## Future Improvements

Future development will focus on improving the model's ability to generalize to real-world oscilloscope images.

Planned improvements include:

- Expand the training dataset
- Add more real oscilloscope photographs
- Implement image augmentation
- Improve waveform-region detection
- Reduce confusion between waveform classes
- Test across different oscilloscope models
- Support noisy and distorted signals
- Improve classification accuracy on real-world images
- Extract additional waveform characteristics

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Convolutional Neural Networks (CNNs)
- Computer Vision
- Image Processing

## Long-Term Goal

The long-term goal is to develop an **AI-assisted oscilloscope analysis system** capable of analyzing an image of an oscilloscope display and automatically determining characteristics such as:

- Waveform Type
- Frequency
- Period
- Amplitude
- RMS Voltage
- Noise
- Harmonic Distortion

This project explores how **machine learning, computer vision, and electrical engineering** can be combined to automate traditional waveform analysis.

## Author

Caden D'Souza
