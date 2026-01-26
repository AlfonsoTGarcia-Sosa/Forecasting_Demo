

### uses python 3.10 ###

import streamlit as st
import pandas as pd
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from torch import nn
from sklearn.metrics import mean_squared_error
import time
from datetime import timedelta
import gc
import plotly.figure_factory as ff
from scipy import stats
import matplotlib.pyplot as plt

# LightGBM (guarded import so your app still runs if it's not installed yet)
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

#from lightgbm import LGBMRegressor


# ---- Reproducibility & determinism setup ----
import os, random

SEED = 42  # pick your favorite

def set_global_seed(seed: int = SEED, deterministic: bool = True):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn / compile-time determinism settings
    if deterministic:
        # Enforce deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # cuDNN settings (only has effect when device.type == 'cuda')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For full cuBLAS determinism on some ops (CUDA only)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Note: MPS (Apple GPU) does not guarantee full determinism across runs.
    # This setup still improves repeatability but tiny jitter can remain.

if "seeded" not in st.session_state:
    set_global_seed(SEED)
    st.session_state.seeded = True

# Call once, as early as possible:
set_global_seed(SEED)

def build_lgbm_supervised(df: pd.DataFrame,
                          date_col: str,
                          target_col: str,
                          exo_cols: list[str],
                          lag_steps: int):
    """
    Returns (X, y) as a single DataFrame with lagged features:
      - target lags: y_{t-1..t-lag_steps}
      - exogenous lags: x_{t}, x_{t-1}, ..., x_{t-min(lag_steps,3)} (short lag set to keep it light)
    """
    df = df.copy()
    # Target lags
    for k in range(1, lag_steps + 1):
        df[f"{target_col}_lag_{k}"] = df[target_col].shift(k)
    # Exogenous lags (0..3 or up to lag_steps if you prefer heavier features)
    max_exo_lag = min(3, lag_steps)
    for col in exo_cols:
        df[f"{col}_lag_0"] = df[col]                # value at t
        for k in range(1, max_exo_lag + 1):
            df[f"{col}_lag_{k}"] = df[col].shift(k)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if any(s in c for s in ["_lag_"])]
    return df[[date_col, target_col] + feature_cols], feature_cols



# First, set the page config (must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="Multi-Step Deep Learning Time Series Forecasting", page_icon="")

# Check if MPS is available (for M1 Macs)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    st.write("Using Apple M1 GPU acceleration")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# st.set_page_config(layout="wide",page_title="Multi-Step Time Series Forecasting LSTM", page_icon="")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
col3, col4 = st.columns([1,5])
col1, col2 = st.columns([3,2])

# Add new session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scalers' not in st.session_state:
    st.session_state.scalers = None
if 'last_sequence' not in st.session_state:
    st.session_state.last_sequence = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()


class LSTMForecasting(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, linear_hidden_size, lstm_num_layers, linear_num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.linear_hidden_size = linear_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.linear_num_layers = linear_num_layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear_layers = nn.ModuleList()
        self.linear_num_layers-=1
        self.linear_layers.append(nn.Linear(self.lstm_hidden_size, self.linear_hidden_size))

        for _ in range(linear_num_layers): 
            self.linear_layers.append(nn.Linear(self.linear_hidden_size, int(self.linear_hidden_size/1.5)))
            self.linear_hidden_size = int(self.linear_hidden_size/1.5)
        
        self.fc = nn.Linear(self.linear_hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0)) 

        # Apply dropout after LSTM
        out = self.dropout(out)

        for linear_layer in self.linear_layers:
            out = linear_layer(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
if 'sd_click' not in st.session_state:
    st.session_state.sd_click = False

if 'train_click' not in st.session_state:
    st.session_state.train_click = False

if 'disable_opt' not in st.session_state:
    st.session_state.disable_opt = False

if 'model_save' not in st.session_state:
    st.session_state.model_save = None

def crps_nrg(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)
    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, weights=sample_weight)

def crps_pwm(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.average(per_obs_crps, weights=sample_weight)

def crps_gaussian(x, mu, sig, sample_weight=None):
    """CRPS for Gaussian distribution"""
    sx = (x - mu) / sig
    pdf = stats.norm.pdf(sx)
    cdf = stats.norm.cdf(sx)
    per_obs_crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - 1. / np.sqrt(np.pi))
    return np.average(per_obs_crps, weights=sample_weight)


#def split_sequences(sequences, n_steps_in, n_steps_out):
#  X, y = list(), list()
#  for i in range(len(sequences)):
#      end_ix = i + n_steps_in
#      out_end_ix = end_ix + n_steps_out#

#      if out_end_ix > len(sequences):
#          break

#      seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
#      X.append(seq_x)
#      y.append(seq_y)
#  return torch.from_numpy(array(X)).float(), torch.from_numpy(array(y)).float()

def make_windows(X_all_scaled: np.ndarray,
                 y_scaled: np.ndarray,
                 n_steps_in: int,
                 n_steps_out: int):
    """
    X_all_scaled: [N, F_exo + 1]  (last column = TARGET as a feature for lags)
    y_scaled:     [N, 1]          (TARGET only)
    Returns:
      X: [num_samples, n_steps_in, F_exo + 1]
      y: [num_samples, n_steps_out]
    """
    X_list, y_list = [], []
    N = len(X_all_scaled)
    for i in range(N):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > N:
            break
        X_list.append(X_all_scaled[i:end_ix, :])
        y_list.append(y_scaled[end_ix:out_end_ix, 0])
    X = torch.from_numpy(np.array(X_list)).float()
    y = torch.from_numpy(np.array(y_list)).float()
    return X, y

def build_lgbm_supervised(df: pd.DataFrame,
                          date_col: str,
                          target_col: str,
                          exo_cols: list,
                          lag_steps: int):
    """
    Returns (supervised_df, feature_cols) where supervised_df contains:
      - target lags: y_{t-1..t-lag_steps}
      - exogenous lags: x_{t}, x_{t-1}, ..., x_{t-min(lag_steps,3)}
    Dropna is applied to remove initial rows without full lag context.
    """
    df = df.copy()

    # Ensure no NaN in original columns before creating lags (robust imputation)
    for col in [target_col] + list(exo_cols):
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').ffill().bfill()

    # Target lags
    for k in range(1, lag_steps + 1):
        df[f"{target_col}_lag_{k}"] = df[target_col].shift(k)

    # Short exogenous lags to keep it light (you can increase to lag_steps if desired)
    max_exo_lag = min(3, lag_steps)
    for col in exo_cols:
        df[f"{col}_lag_0"] = df[col]
        for k in range(1, max_exo_lag + 1):
            df[f"{col}_lag_{k}"] = df[col].shift(k)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.endswith("_lag_0") or "_lag_" in c]
    return df[[date_col, target_col] + feature_cols], feature_cols



def onClickSD():
    st.session_state.sd_click = True

def onClickTrain():
    st.session_state.train_click = True

def update_prediction():
    current_time = time.time()
    if current_time - st.session_state.last_update_time < 0.1:  # 100ms debounce
        return
    
    if st.session_state.model is None:
        return

    try:
        # Get slider values
        new_values = []
        for feature in input_f:
            val = st.session_state[f"slider_{feature}"]
            new_values.append(val)
            # st.write(f"Debug - {feature}: {val}")  # Debug print

        # Create prediction sequence
        pred_sequence = st.session_state.last_sequence.clone()
        scaled_new_values = []
                
        # Scale all input features
        for i, val in enumerate(new_values):
            scaled_val = st.session_state.scalers[i].transform([[val]])[0][0]
            scaled_new_values.append(scaled_val)
            # st.write(f"Debug - Scaled value {i}: {scaled_val}")  # Debug print
        
        # Add target variable's last value (since it's part of the input sequence)
        # target_val = st.session_state.model_save[2].iloc[-1]  # Use last actual value
        # scaled_target = st.session_state.scalers[-1].transform([[target_val]])[0][0]
        # scaled_new_values.append(scaled_target)
        # # st.write(f"Debug - Scaled target value: {scaled_target}")  # Debug print
        # # st.write(f"Debug - Total scaled values: {len(scaled_new_values)}")

        # # Make sure the dimensions match
        # if len(scaled_new_values) != pred_sequence.shape[1]:
        #     st.error(f"Input dimension mismatch. Expected {pred_sequence.shape[1]} features, got {len(scaled_new_values)}")
        #     return

        # pred_sequence[-1, :] = torch.tensor(scaled_new_values)
        # # st.write(f"Debug - Updated sequence shape: {pred_sequence.shape}")
        
# Keep the last observed TARGET (scaled) as the final feature in the row
        target_last_val = st.session_state.model_save[2].iloc[-1]
        scaled_target_last = st.session_state.scalers[-1].transform([[target_last_val]])[0][0]
        scaled_new_values.append(scaled_target_last)  # ensure len == F+1

        # Minimal dimensionality check (restored)
        if len(scaled_new_values) != pred_sequence.shape[1]:
            st.error(
                f"Input dimension mismatch. Expected {pred_sequence.shape[1]} features, "
                f"got {len(scaled_new_values)}"
            )
            return

        pred_sequence[-1, :] = torch.tensor(scaled_new_values, dtype=torch.float32)

        # Make prediction for all forecast steps
        # with torch.no_grad():
        with torch.inference_mode():
            # Move to device and add batch dimension
            pred_sequence = pred_sequence.unsqueeze(0).to(device)
            # st.write(f"Debug - Input to model shape: {pred_sequence.shape}")
            
            # Get prediction
            new_pred = st.session_state.model(pred_sequence)
            # st.write(f"Debug - Model output shape: {new_pred.shape}")
            
            # Convert back to numpy and rescale
            new_pred_unscaled = st.session_state.scalers[-1].inverse_transform(
                new_pred.cpu().numpy().reshape(1, -1)
            )[0]
            
            # st.write(f"Debug - New prediction: {new_pred_unscaled}")  # Debug print

            # Update all predictions while keeping original dates and actual values
            st.session_state.model_save = [
                st.session_state.model_save[0],  # Keep original loss
                new_pred_unscaled,               # Update predictions
                st.session_state.model_save[2],  # Keep actual values
                st.session_state.model_save[3]   # Keep dates
            ]
            
            # Force streamlit to rerun
            st.experimental_rerun()
        
        st.session_state.last_update_time = current_time
    
    except Exception as e:
        st.error(f"Error updating prediction: {str(e)}")
        # st.write("Debug - Full error:", str(e.__class__.__name__), str(e))
        import traceback
        # st.write("Debug - Traceback:", traceback.format_exc())


def update_prediction_lgbm():
    """Update LightGBM predictions when sliders change."""
    current_time = time.time()
    if current_time - st.session_state.last_update_time < 0.1:  # 100ms debounce
        return

    if st.session_state.model is None:
        return

    if st.session_state.get('model_type') != 'LightGBM':
        return

    try:
        # Get the stored input feature names and feature columns
        lgbm_input_features = st.session_state.get('lgbm_input_features', [])
        feat_cols = st.session_state.get('lgbm_feature_cols', [])

        if not lgbm_input_features or not feat_cols:
            return

        # Copy the anchor row (last feature row used for inference)
        X_modified = st.session_state.lgbm_anchor.copy()

        # Update _lag_0 columns with slider values (current values)
        for feature in lgbm_input_features:
            slider_key = f"slider_{feature}"
            lag_0_col = f"{feature}_lag_0"

            if slider_key in st.session_state and lag_0_col in X_modified.columns:
                X_modified[lag_0_col] = st.session_state[slider_key]

        # Get predictions from all H models
        lgbm_models = st.session_state.model
        lgbm_preds = np.array([mdl.predict(X_modified)[0] for mdl in lgbm_models])

        # Update predictions while keeping original dates and actual values
        st.session_state.model_save = [
            st.session_state.model_save[0],  # Keep original loss
            lgbm_preds,                       # Update predictions
            st.session_state.model_save[2],  # Keep actual values
            st.session_state.model_save[3]   # Keep dates
        ]

        st.session_state.last_update_time = current_time

        # Force streamlit to rerun
        st.experimental_rerun()

    except Exception as e:
        st.error(f"Error updating LightGBM prediction: {str(e)}")


def preProcessData(date_f, input_f, output_f):
    preProcessDataList = input_f[:]  # Create a copy to avoid modifying the original
    preProcessDataList.append(output_f)  # Append output_f instead of inserting
    preProcessDF = df[list(dict.fromkeys(preProcessDataList))].copy()  # Use a copy to avoid warnings

    # Clean the data
    for col in preProcessDF.columns:
      preProcessDF[col] = pd.to_numeric(preProcessDF[col], errors='coerce')
    
    preProcessDF = preProcessDF.interpolate(method='linear')
    preProcessDF = preProcessDF.bfill()

    preProcessDF.insert(0, date_f, df[date_f])
    if str(preProcessDF.at[0, date_f]).isdigit():
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f], format='%Y')
    else:
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f])

    return preProcessDF

def check_date_frequency(date_series):
    dates = pd.to_datetime(date_series)
    
    differences = (dates - dates.shift(1)).dropna()
    
    daily_count = (differences == timedelta(days=1)).sum()
    hourly_count = (differences == timedelta(hours=1)).sum()
    weekly_count = (differences == timedelta(weeks=1)).sum()
    monthly_count = (differences >= timedelta(days=28, hours=23, minutes=59)).sum()  # Approximate 28 days to a month
    
    if daily_count > max(monthly_count, hourly_count, weekly_count):
        return 365
    elif monthly_count > max(daily_count, hourly_count, weekly_count):
        return 12
    elif weekly_count > max(daily_count, hourly_count, monthly_count):
        return 52
    elif hourly_count > max(daily_count, weekly_count, monthly_count):
        return 24*365  # Assuming hourly data is daily data repeated every hour
    else:
        return 1

def sea_decomp(date_f,input_f,output_f):
    if date_f:
        sea_decomp_data = preProcessData(date_f,input_f,output_f)
        corr_df = sea_decomp_data.select_dtypes(include=['int', 'float'])
        correlation_matrix = np.round(corr_df.corr(), 1)
        result = seasonal_decompose(sea_decomp_data.set_index(date_f)[output_f], model='additive',period=check_date_frequency(sea_decomp_data[date_f]))
        
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=result.seasonal.index.values, y=result.seasonal.values, mode='lines',line=dict(color='orange')))
        fig_s.update_layout(title='Seasonal',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=300)
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=result.trend.index.values, y=result.trend.values, mode='lines',line=dict(color='orange')))
        fig_t.update_layout(title='Trend',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        height=300)
        
        fig_corr = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale='Viridis')
        
        with st.container(border=True):
            st.subheader("Correlation Matrix:")
            st.divider()
            st.plotly_chart(fig_corr,use_container_width=True)

        with st.container(border=True):
            st.subheader("Seasonal Decompose:")
            st.divider()
            st.plotly_chart(fig_t,use_container_width=True)
            st.divider()
            st.plotly_chart(fig_s,use_container_width=True)

        with st.container(border=True):
            st.subheader("Pre-Processed Data Preview:")
            st.divider()
            st.metric(label="Total Rows:", value=sea_decomp_data.shape[0])
            st.metric(label="Total Columns:", value=sea_decomp_data.shape[1])
            st.dataframe(sea_decomp_data,use_container_width=True,height=250)
        return sea_decomp_data

with col3:
    # st.image("./proekspert-logo-valge.png")
    st.subheader("copyright © 2025 Alfonso T. Garcia-Sosa")


with col4:
    st.title("Multi-Step Deep Learning Time Series Forecasting")
    st.subheader("Operational Parameter Optimization")

        
with col1:

    with st.container(border=True):
        st.subheader("CSV File Uploader:")
        st.divider()
        uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';', decimal=',', skipinitialspace=True)
        except pd.errors.ParserError as e:
            st.error(f"Error parsing the CSV file: {e}")
            st.stop()  # Stop execution if there's an error
        except Exception as e:
          st.error(f"An unexpected error occurred: {e}")
          st.stop()

        df.columns = df.columns.str.replace('-', '_')


        with st.container(border=True):
            st.subheader("Feature Selection:")
            st.divider()
            date_f = st.selectbox(label="Select Date Feature:",options=df.columns,key="date_f1")
            
            input_f = st.multiselect(label="Input Features (X):",options=[element for element in list(df.columns) if element != date_f])

            output_f = st.selectbox(label="Output Feature (Y):",options=[element for element in list(df.columns) if element != date_f])
            st.divider()
            st.button('Pre-Process Data',type="primary",on_click=onClickSD)


        with col2:
            
            with st.container(border=True):
                st.subheader("Dataset Preview:")
                st.divider()
                st.metric(label="Total Rows:", value=df.shape[0])
                st.metric(label="Total Columns:", value=df.shape[1])
                st.dataframe(df,use_container_width=True,height=250)

            if st.session_state.sd_click==True:
                data = sea_decomp(date_f,input_f,output_f)

               
        with st.container(border=True):
            st.subheader("Train Model:")
            st.divider()
            lag_steps = st.number_input('Lag Steps:',step=1,min_value=1)
            forecast_steps = st.number_input('Forecast Steps:',step=1,min_value=1)

            # Forecast start date selector (only show after preprocessing)
            forecast_start_idx = None
            if st.session_state.sd_click and 'data' in dir() or st.session_state.get('preprocessed_data') is not None:
                # Store preprocessed data in session state for access here
                if st.session_state.sd_click and 'data' in dir():
                    st.session_state.preprocessed_data = data

                if st.session_state.get('preprocessed_data') is not None:
                    prep_data = st.session_state.preprocessed_data
                    dates = prep_data[date_f]

                    st.divider()
                    st.subheader("Forecast Start Date:")
                    st.info("Select where forecasting should begin. Training uses data before this date.")

                    # Create a slider with date index
                    min_idx = lag_steps + 1  # Need at least lag_steps rows for training
                    max_idx = len(dates) - forecast_steps  # Leave room for forecast

                    if min_idx < max_idx:
                        forecast_start_idx = st.slider(
                            "Forecast starts at:",
                            min_value=min_idx,
                            max_value=max_idx,
                            value=max_idx,  # Default to end (current behavior)
                            format="%d",
                            key="forecast_start_slider"
                        )
                        selected_date = dates.iloc[forecast_start_idx]
                        st.write(f"**Selected date:** {selected_date}")
                        st.write(f"Training: rows 0-{forecast_start_idx-1} | Forecast: rows {forecast_start_idx}-{forecast_start_idx + forecast_steps - 1}")
                    else:
                        st.warning("Not enough data for selected lag/forecast steps")

            if (lag_steps+forecast_steps)>(df.shape[0]-forecast_steps):
                st.error(f'Lag Steps + Forecast Steps = {lag_steps+forecast_steps} should be <= {df.shape[0]-forecast_steps} (i.e Train set:({lag_steps+forecast_steps}) + Test set:({forecast_steps}) = {lag_steps+forecast_steps+forecast_steps} (>57))', icon="ℹ️")
                st.session_state.disable_opt = True
            else:
                st.session_state.disable_opt = False

            st.divider()

            # Model Selection
            model_choice = st.radio(
                "Select Model:",
                ["LSTM", "LightGBM"] if _HAS_LGBM else ["LSTM"],
                horizontal=True,
                help="Choose which model to train"
            )
            
            st.divider()

            # Conditional parameters based on model choice
            if model_choice == "LSTM":
                lstm_layers = st.slider('LSTM Layers:', 1, 5)
                lstm_neurons = st.slider('LSTM Neurons:', 1, 500)
                st.divider()
                linear_hidden_layers = st.slider('Hidden Layers:', 1, 5)
                linear_hidden_neurons = st.slider('Hidden Neurons:', lstm_neurons,500)
                st.divider()
                n_epochs = st.number_input('No. of Epochs',step=1,min_value=1)
                batch_size = st.number_input('Batch Size',step=1,min_value=1,max_value=100)
                st.divider()
            else:  # LightGBM
                n_estimators = st.slider('Number of Trees:', 50, 1000, 400, step=50)
                learning_rate = st.slider('Learning Rate:', 0.01, 0.3, 0.05, step=0.01)
                num_leaves = st.slider('Number of Leaves:', 10, 100, 31)
                st.divider()
            
            # Add input feature sliders (for both LSTM and LightGBM)
            if input_f:  # Only show if input features are selected
                st.subheader("Input Feature Adjustments")
                st.info("Adjust input features for prediction (optional)")
                
                # Use the same preprocessed data that works for training
                clean_data = preProcessData(date_f, input_f, output_f)
                
                # Create columns for sliders - 2 columns layout
                for i in range(0, len(input_f), 2):
                    col1, col2 = st.columns(2)
                    
                    # Helper function to create slider
                    def create_slider(feature, col):
                        with col:
                            try:
                                min_val = float(clean_data[feature].min())
                                max_val = float(clean_data[feature].max())
                                current_val = float(clean_data[feature].iloc[-1])
                                
                                # Handle edge cases
                                if min_val == max_val:
                                    min_val = min_val - abs(min_val * 0.1) if min_val != 0 else -1.0
                                    max_val = max_val + abs(max_val * 0.1) if max_val != 0 else 1.0
                                
                                # Ensure current_val is within bounds
                                current_val = max(min_val, min(max_val, current_val))
                                
                                # Choose callback based on model type
                                if st.session_state.model is not None:
                                    model_type = st.session_state.get('model_type', 'LSTM')
                                    callback = update_prediction_lgbm if model_type == 'LightGBM' else update_prediction
                                else:
                                    callback = None

                                return st.slider(
                                    f"{feature}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=current_val,
                                    step=0.0001,
                                    key=f"slider_{feature}",
                                    format="%.4f",
                                    on_change=callback
                                )
                            except Exception as e:
                                st.warning(f"Could not create slider for {feature}: {str(e)}")
                                return None
                    
                    # Create sliders for both columns
                    if i < len(input_f):
                        create_slider(input_f[i], col1)
                    if i + 1 < len(input_f):
                        create_slider(input_f[i + 1], col2)
                
                st.divider()

            st.error("Preprocess data before clicking on Train Model")
            sr = st.button(f'Train {model_choice} Model',type="primary",on_click=onClickTrain, disabled=st.session_state.disable_opt)
            st.error('Adjusting features or parameters after training will not maintain the session. Please ensure to retrain after making changes.', icon="ℹ️")


            if sr:
                with st.container(border=True):
                    # Use forecast_start_idx if set, otherwise default to end
                    if forecast_start_idx is not None:
                        df_train = data.iloc[:forecast_start_idx]
                        df_test = data.iloc[forecast_start_idx:forecast_start_idx + forecast_steps]
                    else:
                        df_train = data[:-forecast_steps]
                        df_test = data[-forecast_steps:]

                    if model_choice == "LSTM":
                        # ===== LSTM TRAINING =====
                        # --- SCALE on TRAIN only; keep TARGET as last feature in X, and separately as y ---
                        scalers = []
                        X_feats_scaled = np.empty((df_train.shape[0], 0))

            # Scale exogenous inputs in the UI-selected order
                        for col in input_f:
                            s = MinMaxScaler()
                            vals = s.fit_transform(df_train[col].to_numpy().reshape(-1, 1))
                            X_feats_scaled = np.hstack([X_feats_scaled, vals])
                            scalers.append(s)

            # Scale TARGET once (used both as a feature column for lags and as the label source)
                        target_scaler = MinMaxScaler()
                        y_train_scaled = target_scaler.fit_transform(df_train[output_f].to_numpy().reshape(-1, 1))
                        scalers.append(target_scaler)  # keep LAST (your slider code relies on this ordering)

            # X_all for TRAIN set = exogenous features + TARGET column as last feature
                        X_all_scaled_train = np.hstack([X_feats_scaled, y_train_scaled])

            # Build windows dynamically with current UI selections
                        X, y = make_windows(
                        X_all_scaled_train,
                        y_train_scaled,
                        n_steps_in=int(lag_steps),
                        n_steps_out=int(forecast_steps)
                        )

            # Cache for slider inference
                        st.session_state.last_sequence_train = X[-1].clone()                # [lag_steps, F+1]
                        st.session_state.scalers_train = scalers


                        X = X.to(device)
                        y = y.to(device)
                        dataset = torch.utils.data.TensorDataset(X,y)
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        model = LSTMForecasting(input_size=X.shape[2], lstm_hidden_size=lstm_neurons, lstm_num_layers=lstm_layers, linear_num_layers=linear_hidden_layers,linear_hidden_size=linear_hidden_neurons, output_size=forecast_steps).to(device)
                        st.write(model)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
                        
                        total_marks = n_epochs
                        progress_text = "Training LSTM. Please Wait..."
                        my_bar = st.progress(0, text=progress_text)

                        patience = 3  # Number of epochs to wait for improvement
                        best_loss = float('inf')
                        patience_counter = 0
                        best_model_state = None

                        # Enable cuDNN benchmarking for faster training
                        torch.backends.cudnn.benchmark = True

                        # Use mixed precision for faster training on M1
                        scaler_cuda = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

                        start_time = time.time()

                        # Training loop
                        for epoch in range(1,n_epochs+1):
                            model.train()
                            epoch_loss = 0
                            # After each epoch, clear memory
                            if device.type == 'cuda' or device.type == 'mps':
                                torch.cuda.empty_cache()

                            # Process in larger batches
                            for inputs, labels in dataloader:
                                # Move data to device at the beginning of each batch
                                inputs, labels = inputs.to(device), labels.to(device)
                                
                                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                                if scaler_cuda is not None:  # Use mixed precision if available
                                    with torch.cuda.amp.autocast():
                                        outputs = model(inputs)
                                        loss = criterion(outputs, labels)
                
                                    # Scale loss and perform backward pass
                                    scaler_cuda.scale(loss).backward()
                                    scaler_cuda.step(optimizer)
                                    scaler_cuda.update()
                                else:
                                    # Regular training
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)
                                    loss.backward()
                                    # Gradient clipping to prevent exploding gradients
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                                    optimizer.step()

                                # Accumulate loss for the entire epoch
                                epoch_loss += loss.item() * inputs.size(0)

                            # Calculate average loss for the epoch
                            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
                            
                            # Early stopping check using average epoch loss
                            if avg_epoch_loss < best_loss:
                                best_loss = avg_epoch_loss
                                patience_counter = 0
                                # Save the best model 
                                best_model_state = model.state_dict().copy()   
                            else:
                                patience_counter += 1
                            
                            # Update progress bar
                            my_bar.progress(int((epoch / total_marks) * 100), text=progress_text)
                            
                            # Check if we should stop early
                            if patience_counter >= patience:
                                st.write(f"Early stopping at epoch {epoch}/{n_epochs}")
                                # Load the best model
                                model.load_state_dict(best_model_state)
                                break

                        my_bar.empty()

                        end_time = time.time()
                        training_time = end_time - start_time
                        training_minutes = int(training_time // 60)
                        training_seconds = int(training_time % 60)
                        st.write(f"Training completed in {training_minutes} minutes and {training_seconds} seconds")

                        st.success('Training Completed!', icon="✅")

                        with torch.inference_mode():
        # Use the cached last training window (shape [lag_steps, F+1])
                            X_infer = st.session_state.last_sequence_train.unsqueeze(0).to(device)
                            outputs = model(X_infer)
                            outputs_unscaled = st.session_state.scalers_train[-1].inverse_transform(
                            outputs[0].reshape(1, -1).detach().cpu().numpy()
                            )


                        st.session_state.model = model
                        st.session_state.model_type = "LSTM"
                        st.session_state.scalers = st.session_state.scalers_train
                        st.session_state.last_sequence = st.session_state.last_sequence_train
                        del model; gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        st.session_state.model_save = [
                            f'{loss.item():.4f}',
                            outputs_unscaled[0],
                            df_test[output_f],
                            df_test[date_f]
                        ]

                    else:  # LightGBM Training
                        # ===== LIGHTGBM TRAINING =====
                        try:
                            # Build supervised table
                            supervised_df, feat_cols = build_lgbm_supervised(
                                df=df_train,
                                date_col=date_f,
                                target_col=output_f,
                                exo_cols=input_f,
                                lag_steps=int(lag_steps)
                            )
                            
                            # Split train/test
                            N = len(supervised_df)
                            H = int(forecast_steps)
                            train_end = N - H

                            # Safety check: ensure we have enough data
                            if N == 0:
                                st.error("No data available after preprocessing. Check for missing values in your data.")
                                st.stop()
                            if train_end <= H:
                                st.error(f"Not enough data for training. Have {N} rows after lag creation, need at least {2*H+1} rows. Try reducing Lag Steps or Forecast Steps.")
                                st.stop()

                            X_all = supervised_df[feat_cols]
                            y_all = supervised_df[output_f]

                            X_train_lgbm = X_all.iloc[:train_end]
                            X_anchor = X_all.iloc[train_end-1:train_end]
                            
                            # Progress bar
                            progress_text = "Training LightGBM. Please Wait..."
                            my_bar = st.progress(0, text=progress_text)
                            
                            start_time = time.time()
                            
                            # Train H independent models
                            lgbm_models = []
                            lgbm_preds = []
                            
                            for h in range(1, H+1):
                                y_shift = y_all.shift(-h).iloc[:train_end]
                                X_h = X_train_lgbm.iloc[:len(y_shift)]
                                
                                mdl = LGBMRegressor(
                                    n_estimators=int(n_estimators),
                                    learning_rate=learning_rate,
                                    num_leaves=int(num_leaves),
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                    random_state=SEED,
                                    verbose=-1
                                )
                                mdl.fit(X_h, y_shift)
                                lgbm_pred_h = mdl.predict(X_anchor)[0]
                                lgbm_preds.append(lgbm_pred_h)
                                lgbm_models.append(mdl)
                                
                                # Update progress
                                my_bar.progress(int((h / H) * 100), text=progress_text)
                            
                            my_bar.empty()
                            
                            end_time = time.time()
                            training_time = end_time - start_time
                            training_minutes = int(training_time // 60)
                            training_seconds = int(training_time % 60)
                            st.write(f"Training completed in {training_minutes} minutes and {training_seconds} seconds")
                            
                            st.success('Training Completed!', icon="✅")
                            
                            # Calculate training RMSE (on last model as proxy)
                            train_pred_last = lgbm_models[-1].predict(X_h)
                            train_rmse = np.sqrt(mean_squared_error(y_shift, train_pred_last))
                            
                            # Store in session state
                            st.session_state.model = lgbm_models
                            st.session_state.model_type = "LightGBM"
                            st.session_state.lgbm_feature_cols = feat_cols
                            st.session_state.lgbm_anchor = X_anchor
                            st.session_state.lgbm_input_features = input_f  # Store for slider callback
                            
                            st.session_state.model_save = [
                                f'{train_rmse:.4f}',
                                np.array(lgbm_preds),
                                df_test[output_f],
                                df_test[date_f]
                            ]
                            
                            # Plot feature importance for last model
                            st.subheader("Feature Importance (Last Forecast Step)")
                            fig_imp, ax = plt.subplots(figsize=(10, 6))
                            lgb.plot_importance(lgbm_models[-1], importance_type="gain", ax=ax, max_num_features=15)
                            st.pyplot(fig_imp)
                            plt.close(fig_imp)
                            
                        except Exception as e:
                            st.error(f"LightGBM training failed: {str(e)}")
                            import traceback
                            st.write(traceback.format_exc())


        if st.session_state.train_click==True:
            with st.status('Visualizing Predictions...',expanded=True):
                st.divider()
                # Display metrics
                col1, col2 = st.columns(2)
                col1.metric(label="Training Loss:", value=st.session_state.model_save[0])
                col2.metric(label="Testing RMSE:", value=int(np.sqrt(mean_squared_error(st.session_state.model_save[2], st.session_state.model_save[1]))))
                                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(st.session_state.model_save[2], st.session_state.model_save[1]))

                # Cumulative Ranked Probability Score
                crps_nrg_score = crps_nrg(
                    np.array(st.session_state.model_save[2]).reshape(-1, 1),
                    np.array(st.session_state.model_save[1]).reshape(-1, 1)
                )

                # For Gaussian CRPS, use predictions as mean and calculate uncertainty
                mu = np.array(st.session_state.model_save[1])
                # Estimate uncertainty as rolling standard deviation of predictions
                sig = pd.Series(mu).rolling(window=3, min_periods=1).std().fillna(method='bfill').values
                sig = np.maximum(sig, 0.1)  # Set minimum uncertainty
                
                crps_gaussian_score = crps_gaussian(
                    np.array(st.session_state.model_save[2]),
                    mu,
                    sig
                )

                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(label="RMSE", value=f"{rmse:.3f}")
                col2.metric(label="CRPS (NRG)", value=f"{crps_nrg_score:.3f}")
                col3.metric(label="CRPS (Gaussian)", value=f"{crps_gaussian_score:.3f}")
                
                st.divider()
                fig_pred = go.Figure()

                # Add uncertainty bands (±1 and ±2 sigma)
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.model_save[3],
                    y=mu + 2*sig,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='95% Confidence',
                    fillcolor='rgba(255, 165, 0, 0.1)',
                    fill=None
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.model_save[3],
                    y=mu + sig,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='68% Confidence',
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    fill='tonexty'
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.model_save[3],
                    y=mu - sig,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    fill='tonexty'
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.model_save[3],
                    y=mu - 2*sig,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    fillcolor='rgba(255, 165, 0, 0.1)',
                    fill='tonexty'
                ))
                
                # Add predicted and actual values
                model_type = st.session_state.get('model_type', 'LSTM')
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.model_save[3],
                    y=st.session_state.model_save[1],
                    mode='lines+markers',
                    name=f'{model_type} Predicted',
                    line=dict(color='orange', dash='dash'),
                    marker=dict(size=10)
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.model_save[3],
                    y=st.session_state.model_save[2],
                    mode='lines+markers',
                    name='Actual',
                    marker=dict(size=10)
                ))
                

                fig_pred.update_layout(
                    title='Testing: Actual vs Predicted with Uncertainty',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=410
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                

                st.metric(label="Cumulative Ranked Probability Score (CRPS) (NRG):", value=f"{crps_nrg_score:.3f}")


                # Add download button for the plot
                img_bytes = fig_pred.to_image(format="png", scale=2, engine="kaleido")
                st.download_button(
                    label="Download Forecast as IMG",
                    data=img_bytes,
                    file_name='forecast_with_uncertainty.png',
                    mime='image/jpeg'
                )

    else:
        st.session_state.sd_click = False

ft = """
<style>
a:link , a:visited{
color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 10vh;
}

footer{
    visibility:hidden;
}

.footer {
position: relative;
left: 0;
top:230px;
bottom: 0;
width: 100%;
background-color: transparent;
color: #808080; 
text-align: center;
padding: 0px 0px 15px 0px; 
}
</style>

<div id="page-container">



</div>
"""
st.write(ft, unsafe_allow_html=True)
