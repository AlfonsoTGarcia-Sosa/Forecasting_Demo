# 📈 Time Series Forecasting App

Multi-step forecasting with **LSTM** and **LightGBM** 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://benford-9xpqy4twevxdn2qlbckcfl.streamlit.app/)

Multi-Step Deep Learning Time Series Forecasting Demo

A Streamlit application for time series forecasting with operational parameter optimization.

Overview

This tool allows you to upload time series data and train either an LSTM or LightGBM model to generate multi-step forecasts with uncertainty quantification.

Runs online at: https://benford-9xpqy4twevxdn2qlbckcfl.streamlit.app/

---

## Features

- 📊 Upload CSV data & automatic preprocessing
- 🔍 Correlation matrix & seasonal decomposition
- 🧠 LSTM (deep learning) or LightGBM (gradient boosting)
- 📉 Multi-step ahead forecasting with uncertainty bands
- 📏 Metrics: RMSE, CRPS

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run LSTM_LightGBM_demo_ATGS.py
```

---

## Quick Usage

1. Upload CSV (`;` separator, `,` decimal)
2. Select date, input features, and target
3. Click **Pre-Process Data**
4. Configure model parameters
5. Click **Train Model** → view predictions

---


<img width="377" height="233" alt="Screenshot 2025-12-30 at 19 10 25" src="https://github.com/user-attachments/assets/fc0fd79b-0498-41ce-a194-a07d74303a01" />


<img width="1131" height="670" alt="Screenshot 2025-12-31 at 22 43 44" src="https://github.com/user-attachments/assets/031a8454-ed1e-41f8-b881-b0372c4a9040" />


<img width="663" height="848" alt="Screenshot 2025-12-31 at 22 50 53" src="https://github.com/user-attachments/assets/5d10d069-5c57-400d-8ae6-eec6b42ceb29" />


<img width="663" height="600" alt="Screenshot 2025-12-31 at 22 55 14" src="https://github.com/user-attachments/assets/ef6654df-a361-4f4f-a9a2-ced3ccf3719f" />



<img width="663" height="331" alt="Screenshot 2025-12-31 at 22 46 52" src="https://github.com/user-attachments/assets/a04d007a-7262-4083-8de4-3ae4a78bb719" />


<img width="663" height="631" alt="Screenshot 2025-12-31 at 22 52 26" src="https://github.com/user-attachments/assets/75adff2b-eb65-444b-a110-fe3c9e759d0c" />


## Requirements

(see requirements.txt)

streamlit, pandas, numpy, torch, scikit-learn
plotly, statsmodels, scipy, lightgbm, matplotlib, kaleido


Feature Description
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

Start

Upload CSV — Use semicolon ; as separator and comma , for decimals
Select Features — Choose date column, input features (X), and target (Y)
Pre-Process Data — Click to view correlation matrix and seasonal decomposition
Configure Model — Set lag steps, forecast steps, and model hyperparameters
Train Model — Click to train and view predictions with uncertainty bands


## ⚙️ Model Parameters

| LSTM | LightGBM |
|------|----------|
| Lag/Forecast steps | Lag/Forecast steps |
| LSTM layers (1-5) | Number of trees (50-1000) |
| LSTM neurons (1-500) | Learning rate (0.01-0.3) |
| Hidden layers (1-5) | Number of leaves (10-100) |
| Epochs, Batch size | — |

---

## 📊 Sample Data Format

```csv
Date;Temperature;Humidity;Pressure;Target
2024-01-01;22,5;65,0;1013,2;100,5
2024-01-02;23,1;63,5;1012,8;102,3
...
```

---

RMSE — Root Mean Squared Error between predicted and actual values
CRPS (NRG) — Continuous Ranked Probability Score (energy form)
CRPS (Gaussian) — CRPS assuming Gaussian uncertainty

Notes
GPU acceleration supported (CUDA / Apple MPS)
Early stopping enabled for LSTM (patience = 3 epochs)
Reproducibility ensured with fixed seed (42)
Input feature sliders available for LSTM model after training

# Built for operational parameter optimization in time series forecasting.
 
References
 
[LSTM Neural Networks for Time Series Forecasting](https://machinelearningmastery.com/lstm-neural-networks-for-time-series-forecasting/)
[Cumulative Ranked Probability Score](https://www.lokad.com/continuous-ranked-probability-score)
[Seasonal Decomposition of Time Series](https://otexts.com/fpp2/decomposition.html)

# Support
 
For issues, feature requests, or additional training, please contact Alfonso T. Garcia-Sosa t.alfonso@gmail.com


