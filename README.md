# simona-anomalydetection-gnn

## Anomaly Detection and Signal Processing with GNNs and ASLS

Smart Innovation of Stress Monitoring in the Progressive Geotechnical Applications Using Fiber Optic System. Source codes repository of framework on detrending sequential sensor data and anomaly detection.

This project provides a framework for removing trend and detecting anomalies in time-series signals using Graph Neural Network (GNN). It includes functionality for generating synthetic signal datasets, training models, and evaluating performance using outlier detection methods like Local Outlier Factor (LOF).

## Overview

The core objective of this project is to identify anomalies and detect trends in time-series signals. The project includes:

- **Signal Generation**: Generator synthetic nonstationary signals with varying trends, noise, and anomalies (peaks).
- **Trend Estimation**: Trend estiamtor of nonstationary signals. 
- **Anomaly Detection**: Utilization of anomaly detection on detrened signal data.
- **Model Evaluation**: An evaluator the detrending and anomaly detection framework.

## Files and Directories

- `main.py`: Contains functions to manage training and evaluation of algorithms.
- `data/generate_dataset.py`: Contains a class for generating synthetic signals with various trends, noise, and anomalies (e.g., peaks).
- `src/model/GNNModel.py`: Defines the architecture of a Graph Neural Network (GNN) model.
- `src/evaluator.py, src/results_visualiser.py`: Contains the classes for evaluating models and visualizing results. 



