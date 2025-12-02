Multi-Step Deep Learning Time Series Forecasting Demo
A Streamlit application for time series forecasting with operational parameter optimization.
Overview

This tool allows you to upload time series data and train either an LSTM or LightGBM model to generate multi-step forecasts with uncertainty quantification.
Features

Feature
Description
Data Upload 
CSV files (semicolon-separated, comma decimal) 
Preprocessing 
Automatic cleaning, interpolation, and missing value handling 
Visualization 
Correlation matrix, seasonal decomposition (trend & seasonality) 
Models 
LSTM (deep learning) or LightGBM (gradient boosting) 
Metrics 
RMSE, CRPS (NRG), CRPS (Gaussian) 
Export 
Download forecast plot as PNG 
Quick Start

Upload CSV — Use semicolon ; as separator and comma , for decimals
Select Features — Choose date column, input features (X), and target (Y)
Pre-Process Data — Click to view correlation matrix and seasonal decomposition
Configure Model — Set lag steps, forecast steps, and model hyperparameters
Train Model — Click to train and view predictions with uncertainty bands
Model Parameters

LSTM

Parameter
Description
Range
Lag Steps 
Historical window size 
≥ 1 
Forecast Steps 
Prediction horizon 
≥ 1 
LSTM Layers 
Number of LSTM layers 
1–5 
LSTM Neurons 
Hidden units per layer 
1–500 
Hidden Layers 
Dense layers after LSTM 
1–5 
Epochs 
Training iterations 
≥ 1 
Batch Size 
Samples per update 
1–100 
LightGBM

Parameter
Description
Default
Number of Trees 
Boosting iterations 
400 
Learning Rate 
Step size shrinkage 
0.05 
Number of Leaves 
Tree complexity 
31 
Output Metrics

RMSE — Root Mean Squared Error between predicted and actual values
CRPS (NRG) — Continuous Ranked Probability Score (energy form)
CRPS (Gaussian) — CRPS assuming Gaussian uncertainty
Notes

GPU acceleration supported (CUDA / Apple MPS)
Early stopping enabled for LSTM (patience = 3 epochs)
Reproducibility ensured with fixed seed (42)
Input feature sliders available for LSTM model after training
Requirements




streamlit
pandas
numpy
torch
scikit-learn
plotly
statsmodels
scipy
lightgbm (optional)
kaleido (for image export)
Built for operational parameter optimization in time series forecasting.
 
References
 
[LSTM Neural Networks for Time Series Forecasting](https://machinelearningmastery.com/lstm-neural-networks-for-time-series-forecasting/)
[Cumulative Ranked Probability Score](https://www.lokad.com/continuous-ranked-probability-score)
[Seasonal Decomposition of Time Series](https://otexts.com/fpp2/decomposition.html)
 
Support
 
For issues, feature requests, or additional training, please contact Alfonso Garcia-Sosa alfonsog@proekspert.com