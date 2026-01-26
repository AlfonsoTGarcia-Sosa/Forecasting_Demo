# Forecasting_Demo - Technical Documentation

## Overview

A Streamlit application for multi-step time series forecasting using LSTM (Long Short-Term Memory) neural networks and LightGBM (Gradient Boosting) models. Designed for operational parameter optimization in industrial settings (heat pumps, digital twins, predictive maintenance).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT APPLICATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  CSV Upload  │───▶│ Preprocessing │───▶│  Seasonal    │                  │
│  │              │    │  & Cleaning   │    │  Decompose   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                              │                    │                         │
│                              ▼                    ▼                         │
│                     ┌──────────────┐    ┌──────────────┐                   │
│                     │ Correlation  │    │   Trend &    │                   │
│                     │   Matrix     │    │  Seasonality │                   │
│                     └──────────────┘    └──────────────┘                   │
│                              │                                              │
│         ┌────────────────────┴────────────────────┐                        │
│         ▼                                         ▼                        │
│  ┌──────────────┐                         ┌──────────────┐                 │
│  │    LSTM      │                         │   LightGBM   │                 │
│  │   Model      │                         │    Model     │                 │
│  └──────────────┘                         └──────────────┘                 │
│         │                                         │                        │
│         └────────────────────┬────────────────────┘                        │
│                              ▼                                              │
│                     ┌──────────────┐                                       │
│                     │  Forecast &  │                                       │
│                     │ Visualization│                                       │
│                     └──────────────┘                                       │
│                              │                                              │
│                              ▼                                              │
│                     ┌──────────────┐                                       │
│                     │   Sliders    │◀──── What-if Analysis                 │
│                     │  (Feature    │                                       │
│                     │  Adjustment) │                                       │
│                     └──────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow & Windowing

### Sliding Window Approach

Time series data is transformed into supervised learning format using sliding windows:

```
Raw Time Series:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │ t8 │ t9 │t10 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

With lag_steps=3, forecast_steps=2:

Window 1:  [t0, t1, t2] ──▶ predict [t3, t4]
Window 2:  [t1, t2, t3] ──▶ predict [t4, t5]
Window 3:  [t2, t3, t4] ──▶ predict [t5, t6]
   ...
```

### Feature Construction

**For LSTM:**
```
Input tensor shape: [batch_size, lag_steps, num_features + 1]
                                              └── target as feature

┌─────────────────────────────────────────────────────────┐
│                    Input Sequence                        │
├─────────┬─────────┬─────────┬─────────┬────────────────┤
│  Time   │ Feature │ Feature │   ...   │    Target      │
│  Step   │    1    │    2    │         │  (as feature)  │
├─────────┼─────────┼─────────┼─────────┼────────────────┤
│  t-3    │  x1_t-3 │  x2_t-3 │   ...   │    y_t-3       │
│  t-2    │  x1_t-2 │  x2_t-2 │   ...   │    y_t-2       │
│  t-1    │  x1_t-1 │  x2_t-1 │   ...   │    y_t-1       │
└─────────┴─────────┴─────────┴─────────┴────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   LSTM Model    │
                    └─────────────────┘
                              │
                              ▼
              Output: [y_t, y_t+1, ..., y_t+H-1]
```

**For LightGBM:**
```
Flat feature table with lagged columns:

┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ y_lag_1  │ y_lag_2  │ y_lag_3  │ x1_lag_0 │ x1_lag_1 │ x1_lag_2 │ ...
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│  y_t-1   │  y_t-2   │  y_t-3   │  x1_t    │  x1_t-1  │  x1_t-2  │ ...
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Model 1 │     │ Model 2 │     │ Model H │
        │ (h=1)   │     │ (h=2)   │ ... │ (h=H)   │
        └─────────┘     └─────────┘     └─────────┘
              │               │               │
              ▼               ▼               ▼
            y_t+1          y_t+2           y_t+H
```

---

## Model Architectures

### LSTM Neural Network

```
┌─────────────────────────────────────────────────────────────────┐
│                      LSTMForecasting Model                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: [batch, lag_steps, features]                           │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────┐                       │
│   │           LSTM Layer(s)              │                       │
│   │   hidden_size = lstm_neurons         │                       │
│   │   num_layers = lstm_layers           │                       │
│   └─────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────┐                       │
│   │         Dropout (0.2)                │                       │
│   └─────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────┐                       │
│   │      Linear Hidden Layers            │                       │
│   │   (progressively shrinking by 1.5x)  │                       │
│   └─────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────┐                       │
│   │         Dropout (0.2)                │                       │
│   └─────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│   ┌─────────────────────────────────────┐                       │
│   │    Final Linear (→ forecast_steps)   │                       │
│   └─────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│   Output: [batch, forecast_steps]                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LightGBM Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                   LightGBM Multi-Horizon                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   For each horizon h = 1 to H:                                  │
│                                                                  │
│   Training target: y shifted by -h steps                        │
│                                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │        LGBMRegressor(h=1)            │──▶ predicts y_{t+1}  │
│   │   n_estimators, learning_rate,       │                       │
│   │   num_leaves, subsample              │                       │
│   └─────────────────────────────────────┘                       │
│                                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │        LGBMRegressor(h=2)            │──▶ predicts y_{t+2}  │
│   └─────────────────────────────────────┘                       │
│                                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │             ...                      │                       │
│   └─────────────────────────────────────┘                       │
│                                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │        LGBMRegressor(h=H)            │──▶ predicts y_{t+H}  │
│   └─────────────────────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Slider Functionality (What-If Analysis)

### Concept

Sliders enable interactive scenario exploration by modifying input feature values:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLIDER MECHANISM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Original Data                  Modified via Slider             │
│   ┌─────────────────┐           ┌─────────────────┐             │
│   │ Temperature: 45 │  ──────▶  │ Temperature: 52 │             │
│   │ Pressure: 100   │  slider   │ Pressure: 100   │             │
│   │ Flow: 25        │           │ Flow: 25        │             │
│   └─────────────────┘           └─────────────────┘             │
│            │                             │                       │
│            ▼                             ▼                       │
│   ┌─────────────────┐           ┌─────────────────┐             │
│   │     Model       │           │     Model       │             │
│   └─────────────────┘           └─────────────────┘             │
│            │                             │                       │
│            ▼                             ▼                       │
│   Forecast: [50, 48, 47]        Forecast: [58, 55, 52]          │
│                                                                  │
│   "What happens if temperature increases by 7 degrees?"         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LSTM Slider Implementation

```python
# Slider modifies the LAST timestep of the lag sequence
pred_sequence[-1, :] = [scaled_slider_values + scaled_target]
                                     │
                                     ▼
                        ┌─────────────────────┐
                        │   model(sequence)   │
                        └─────────────────────┘
                                     │
                                     ▼
                        inverse_transform(output)
```

### LightGBM Slider Implementation

```python
# Slider modifies _lag_0 columns (current values) in anchor row
X_modified[f"{feature}_lag_0"] = slider_value
                                     │
                                     ▼
            ┌────────────────────────────────────────┐
            │  [model.predict(X_modified) for model] │
            └────────────────────────────────────────┘
                                     │
                                     ▼
                         predictions array
```

**Key Difference:** LightGBM doesn't need scaling (tree-based models are scale-invariant).

---

## Forecast Start Date Selection

### Problem Solved

```
Default behavior (forecast from end):

Data:     Oct 1 ═══════════════════════════════ Oct 31
                        Training                │ Forecast
                                               ▲
                                    Always at the end

With start date slider:

Data:     Oct 1 ═══════════ Oct 26 ════════════ Oct 31
              Training      │    Forecast
                           ▲
                   User selects this point
```

### Implementation

```python
if forecast_start_idx is not None:
    df_train = data.iloc[:forecast_start_idx]           # Before selected date
    df_test = data.iloc[forecast_start_idx:forecast_start_idx + forecast_steps]
else:
    df_train = data[:-forecast_steps]                   # Original behavior
    df_test = data[-forecast_steps:]
```

---

## Evaluation Metrics

### RMSE (Root Mean Square Error)

```
RMSE = √(1/n × Σ(y_actual - y_predicted)²)
```

### CRPS (Continuous Ranked Probability Score)

Measures calibration of probabilistic forecasts:

```
CRPS = E|Y - y| - ½E|Y - Y'|

where Y, Y' are independent copies of the forecast distribution
```

**NRG (Energy) formulation:**
```python
absolute_error = mean(|y_pred - y_true|)
diff = sorted(y_pred)[1:] - sorted(y_pred)[:-1]
weight = arange(1, n) * arange(n-1, 0, -1)
crps = absolute_error - sum(diff * weight) / n²
```

**Gaussian formulation:**
```python
sx = (x - mu) / sig
crps = sig * (sx * (2*CDF(sx) - 1) + 2*PDF(sx) - 1/√π)
```

---

## Data Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw CSV                                                        │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────┐                       │
│   │  pd.read_csv(sep=';', decimal=',')  │  European format      │
│   └─────────────────────────────────────┘                       │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────┐                       │
│   │  pd.to_numeric(errors='coerce')     │  Handle non-numeric   │
│   └─────────────────────────────────────┘                       │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────┐                       │
│   │  interpolate(method='linear')       │  Fill gaps            │
│   │  .ffill().bfill()                   │  Edge cases           │
│   └─────────────────────────────────────┘                       │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────┐                       │
│   │  pd.to_datetime(dayfirst=True)      │  Parse dates          │
│   └─────────────────────────────────────┘                       │
│       │                                                          │
│       ▼                                                          │
│   Clean DataFrame ready for modeling                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Session State Management

```python
st.session_state = {
    # Models
    'model': trained_model,              # LSTM model or list of LightGBM models
    'model_type': 'LSTM' | 'LightGBM',

    # LSTM-specific
    'scalers': [scaler_feat1, ..., scaler_target],
    'last_sequence': tensor[lag_steps, features+1],

    # LightGBM-specific
    'lgbm_feature_cols': ['y_lag_1', 'x1_lag_0', ...],
    'lgbm_anchor': DataFrame[1, num_features],
    'lgbm_input_features': ['x1', 'x2', ...],

    # Results
    'model_save': [loss, predictions, actuals, dates],

    # UI state
    'sd_click': bool,      # Preprocessing done
    'train_click': bool,   # Training done
    'preprocessed_data': DataFrame,
}
```

---

## Files Structure

```
Forecasting_Demo/
├── LSTM_LightGBM_demo_ATGS.py    # Main application
├── requirements.txt               # Dependencies
├── runtime.txt                    # Python version for Streamlit Cloud
├── README.md                      # GitHub readme
└── FOR_ALFONSO.md                 # This documentation
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| pandas | Data manipulation |
| numpy | Numerical operations |
| torch | LSTM neural network |
| lightgbm | Gradient boosting models |
| scikit-learn | MinMaxScaler, metrics |
| plotly | Interactive visualizations |
| statsmodels | Seasonal decomposition |
| scipy | Statistical functions (CRPS) |
| kaleido | Export plots as images |
| matplotlib | Feature importance plots |

---

## Commands Reference

```bash
# Local development
conda activate miPyth3.10
streamlit run LSTM_LightGBM_demo_ATGS.py

# Git workflow
git add <file>
git commit -m "message"
git push origin main

# If push rejected
git pull --rebase origin main
git push origin main
```

---

## Future Enhancements

- [ ] Continuous forecasting (auto-refresh every N minutes)
- [ ] Click on Trend plot to select forecast start date
- [ ] Real-time data streaming integration
- [ ] Model comparison dashboard
- [ ] Export model weights

---

*copyright © 2026 Alfonso T. Garcia-Sosa*
