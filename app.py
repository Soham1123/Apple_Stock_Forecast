# app.py (ARIMA-based Streamlit forecasting app)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Apple Stock Predictor (ARIMA)", layout="wide")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_and_prepare(file_buffer):
    """
    Load Excel/CSV file, aggregate to daily close (if intraday),
    and return DataFrame with columns ['ds', 'y'] (ds = datetime, y = price).
    """
    if file_buffer is None:
        raise FileNotFoundError("No file provided. Upload an Excel/CSV with a timestamp and stock_price columns.")
    # Detect file type and read
    if hasattr(file_buffer, "name") and file_buffer.name.lower().endswith((".xls", ".xlsx")):
        raw = pd.read_excel(file_buffer)
    else:
        raw = pd.read_csv(file_buffer)
    # Heuristics for columns
    ts_col = None
    price_col = None
    for c in raw.columns:
        if 'time' in c.lower() or 'date' in c.lower():
            ts_col = c
            break
    for c in raw.columns:
        if 'price' in c.lower() or 'close' in c.lower() or 'stock' in c.lower():
            price_col = c
            break
    if ts_col is None or price_col is None:
        raise ValueError("Could not detect timestamp or price column. Ensure file has columns with 'date'/'timestamp' and 'price'/'stock_price'.")
    raw[ts_col] = pd.to_datetime(raw[ts_col])
    # Keep only weekdays (safe default)
    raw = raw[raw[ts_col].dt.weekday < 5]
    # Aggregate intraday to daily close (last observation per day)
    raw = raw.sort_values(ts_col)
    daily = raw.groupby(raw[ts_col].dt.date).agg({
        ts_col: 'last',
        price_col: 'last'
    }).reset_index(drop=True)
    daily = daily.rename(columns={ts_col: 'ds', price_col: 'y'})
    daily['ds'] = pd.to_datetime(daily['ds']).dt.normalize()
    # Fill small gaps by forward/back fill; keep as non-stationary original series for plotting
    daily = daily.set_index('ds').asfreq('D')  # calendar days
    daily['y'] = daily['y'].ffill().bfill()
    daily = daily.reset_index()
    return daily

@st.cache_resource
def fit_arima(train_series, order=(1,1,1), maxiter=50):
    """Fit ARIMA and return fitted result (statsmodels ARIMAResults)."""
    model = ARIMA(train_series, order=order)
    res = model.fit(method_kwargs={"maxiter": maxiter})
    return res

# ---------------------------
# App UI
# ---------------------------
st.title("🍎 Apple Stock Predictor — ARIMA (enter a date to get forecast)")

st.sidebar.header("Upload & Model Options")
uploaded = st.sidebar.file_uploader("Upload Excel/CSV (timestamp and stock_price columns)", type=['csv','xlsx','xls'])
use_sample = st.sidebar.checkbox("Use sample demo dataset (if no upload)", value=False)

# ARIMA order controls
p = st.sidebar.number_input("AR (p)", value=1, min_value=0, max_value=5, step=1)
d = st.sidebar.number_input("Diff (d)", value=1, min_value=0, max_value=2, step=1)
q = st.sidebar.number_input("MA (q)", value=1, min_value=0, max_value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("*Graph filters*")

# ---------------------------
# Load data
# ---------------------------
try:
    if use_sample and uploaded is None:
        rng = pd.date_range("2020-01-01", periods=1000, freq='D')
        np.random.seed(0)
        y = np.cumsum(np.random.randn(len(rng)) * 2 + 0.1) + 150
        sample = pd.DataFrame({'ds': rng, 'y': y})
        data = sample
    else:
        data = load_and_prepare(uploaded)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# show data summary
st.subheader("Data summary")
st.write(f"Rows: {len(data)} — date range: {data['ds'].min().date()} to {data['ds'].max().date()}")
st.dataframe(data.tail(6))

# Graph year/month filters
years = sorted(data['ds'].dt.year.unique().tolist())
months = list(range(1,13))
sel_years = st.sidebar.multiselect("Select years to show on graph", years, default=years)
sel_months = st.sidebar.multiselect("Select months (1-12) to show", months, default=months)

# ---------------------------
# Fit ARIMA model (cached)
# ---------------------------
st.info("Fitting ARIMA model to full historical series (this may take a moment)...")
order = (int(p), int(d), int(q))
with st.spinner("Training ARIMA..."):
    series = data.set_index('ds')['y']
    try:
        res = fit_arima(series, order=order, maxiter=200)
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.stop()

st.success("Model fitted ✅")

# Optionally show model summary
if st.checkbox("Show model summary (text)"):
    st.text(res.summary().as_text())

# ---------------------------
# Inputs for prediction
# ---------------------------
st.subheader("Prediction input")
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Enter date (prediction target)", value=(data['ds'].max().date() + timedelta(days=1)))
with col2:
    user_price = st.number_input("Enter a stock price value to compare (optional)", value=0.0, format="%.4f")

input_dt = pd.to_datetime(input_date).normalize()
last_date = data['ds'].max()

if input_dt <= last_date:
    observed_row = data.loc[data['ds'] == input_dt]
    if not observed_row.empty:
        observed_val = float(observed_row['y'].iloc[0])
    else:
        observed_val = np.nan
    st.write(f"Selected date is within historical range. Observed value (if present): *{observed_val}*")
else:
    steps_full = (input_dt - last_date).days
    steps = max(1, steps_full)
    with st.spinner(f"Forecasting {steps} steps ahead to {input_dt.date()} ..."):
        try:
            forecast_obj = res.get_forecast(steps=steps)
            pred_mean = forecast_obj.predicted_mean.iloc[-1]
            pred_ci = forecast_obj.conf_int(alpha=0.05).iloc[-1] if forecast_obj is not None else None
            pred_lower, pred_upper = (float(pred_ci.iloc[0]), float(pred_ci.iloc[1])) if pred_ci is not None else (np.nan, np.nan)
            st.markdown(f"### Forecast for *{input_dt.date()}*")
            st.write(f"*Predicted stock price (mean):* {pred_mean:.4f}")
            st.write(f"*95% prediction interval:* [{pred_lower:.4f}, {pred_upper:.4f}]")
            if user_price and user_price > 0:
                st.write(f"Your entered comparison price: *{user_price:.4f}*")
                st.write(f"Difference (entered - predicted): *{user_price - pred_mean:.4f}*")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# ---------------------------
# NEW Plot block: Day-by-day predictions for each selected month
# ---------------------------
st.subheader("Day-by-day predictions for selected months")

# If no months/years selected, default to recent month
if not sel_years:
    sel_years = [data['ds'].dt.year.max()]
if not sel_months:
    sel_months = [data['ds'].dt.month.max()]

# Build list of calendar days across selected year-month pairs
selected_days = []
for y in sel_years:
    for m in sel_months:
        try:
            start_dt = pd.Timestamp(year=int(y), month=int(m), day=1)
        except Exception:
            continue
        end_dt = (start_dt + pd.offsets.MonthEnd(0)).normalize()
        days_in_month = pd.date_range(start=start_dt, end=end_dt, freq='D')
        selected_days.extend(days_in_month)

if len(selected_days) == 0:
    st.warning("No valid selected month-year combinations. Displaying full series instead.")
    plot_start = data['ds'].min()
    plot_end = data['ds'].max()
    plot_index = pd.date_range(start=plot_start, end=plot_end, freq='D')
else:
    plot_start = min(selected_days) - pd.Timedelta(days=3)
    plot_end = max(selected_days) + pd.Timedelta(days=3)
    plot_index = pd.date_range(start=plot_start, end=plot_end, freq='D')

observed_series = data.set_index('ds')['y'].reindex(plot_index)

# Use ARIMA res to get day-by-day predictions
try:
    pred = res.get_prediction(start=plot_start, end=plot_end)
    pred_mean = pd.Series(pred.predicted_mean.values, index=plot_index)
    pred_ci = pred.conf_int(alpha=0.05)
    lower = pd.Series(pred_ci.iloc[:, 0].values, index=plot_index)
    upper = pd.Series(pred_ci.iloc[:, 1].values, index=plot_index)
except Exception as e:
    st.warning(f"res.get_prediction failed for the chosen range ({e}). Falling back to future-only forecast.")
    pred_mean = pd.Series(index=plot_index, dtype=float)
    lower = pd.Series(index=plot_index, dtype=float)
    upper = pd.Series(index=plot_index, dtype=float)
    if plot_end > last_date:
        steps = (plot_end - last_date).days
        steps = max(1, steps)
        fc = res.get_forecast(steps=steps)
        fc_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        pred_mean.loc[fc_index] = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.05)
        lower.loc[fc_index] = ci.iloc[:,0].values
        upper.loc[fc_index] = ci.iloc[:,1].values

# Highlight selected days in the plot
plot_index_np = np.array(plot_index)
selected_mask = np.isin(plot_index_np, np.array(selected_days).astype('datetime64[D]'))

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(plot_index, observed_series.values, label="Observed (non-stationary series)", color='tab:blue')
ax.plot(plot_index, pred_mean.values, linestyle='--', label='ARIMA predicted (day-by-day)', color='tab:orange')
ax.fill_between(plot_index, lower.values, upper.values, color='orange', alpha=0.2, label='95% CI')
ax.axvline(last_date, color='gray', linestyle=':', label='Last observed date')

if selected_mask.any():
    ax.scatter(plot_index[selected_mask], pred_mean.values[selected_mask], color='red', s=20, alpha=0.8, label='Predicted (selected days)')

ax.set_xlabel("Date")
ax.set_ylabel("Stock price")
ax.set_title("Day-by-day predictions for selected months (with observed series)")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# Optional evaluation on last 30 days
# ---------------------------
if st.checkbox("Evaluate on last 30 days (if present)"):
    if len(data) >= 60:
        y_test = data.set_index('ds')['y'].iloc[-30:]
        f = res.get_forecast(steps=30)
        y_pred = f.predicted_mean.values[:len(y_test)]
        mae = mean_absolute_error(y_test.values, y_pred)
        st.write(f"MAE on last 30 days: *{mae:.4f}*")
    else:
        st.write("Not enough data to evaluate last 30 days.")

# Footer instructions
st.markdown("---")
st.markdown("Instructions: Upload dataset (timestamp + price). Select months/years in sidebar — the plot will show day-by-day predictions for every day in the selected months.")