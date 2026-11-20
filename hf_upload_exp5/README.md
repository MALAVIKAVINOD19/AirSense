# AirSense Experiment 5 – Transformer-Based Air Quality Forecasting

## 1. Project Overview

AirSense Experiment 5 is a Transformer-based time-series forecasting model developed using PyTorch.

The model uses historical air-quality measurements to predict the next-hour readings of nine selected air-quality features.

The main objective is to explore how Transformer architectures can learn patterns from historical air-quality data and generate future numerical predictions.

## 2. Model Architecture

- Model: AirSenseTransformerExp5
- Architecture: Transformer Encoder
- Framework: PyTorch
- Input sequence length: 12 hours
- Input features: 9
- Output features: 9
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Feed-forward dimension: 256
- Dropout: 0.2
- Parameters: 110,857

## 3. Selected Features

The model uses the following nine features:

1. CO(GT)
2. PT08.S1(CO)
3. C6H6(GT)
4. PT08.S2(NMHC)
5. NOx(GT)
6. PT08.S3(NOx)
7. NO2(GT)
8. PT08.S4(NO2)
9. PT08.S5(O3)

## 4. How the Model Works

1. Receives 12 hours of historical air-quality readings.
2. Normalizes the input using the training dataset's mean and standard deviation.
3. Processes the input through the Transformer encoder.
4. Generates predictions for all nine selected features.
5. Converts predictions back into their original measurement scales.

## 5. Files

- `airsense_exp5_best.pt` – Trained model checkpoint.
- `model.py` – Transformer architecture.
- `config.json` – Model configuration.
- `mean.npy` – Training normalization means.
- `std.npy` – Training normalization standard deviations.
- `requirements.txt` – Required Python libraries.

## 6. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 7. Intended Use

This model is developed for experimental air-quality time-series forecasting and educational purposes.

It is intended to predict the next-hour values of selected air-quality features based on historical measurements.

## 8. Limitations

- Predictions are estimates, not actual sensor measurements.
- The model was developed using a specific air-quality dataset.
- Performance on new locations and real-world sensor data has not been established.
- The model does not provide medical diagnoses or guarantee accurate future predictions.

## 9. Project

**Project Name:** AirSense

**Experiment:** 5 – Feature Selection Transformer

**Framework:** PyTorch