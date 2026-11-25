"""Core logic for the Community Microgrid Energy Forecasting project.

Pure functions (no Streamlit imports) so they can be unit-tested.
Sections: synthetic data (Report 7.1), features/models (6.5, 7.2),
dispatch + alerts (7.4) and edge-side logic (6.4, 8.2).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
START_DATE = "2026-08-01"
DAYS = 90
TRAIN_DAYS = 72          # chronological split: first 72 days train,
TEST_DAYS = 18           # last 18 days (432 h) test
PV_KWP = 120
SYSTEM_EFFICIENCY = 0.82
TEMP_DERATE = 0.004      # per deg C above 25 C

# Battery used in the dispatch simulation (Report 7.4)
BATT_KWH, BATT_KW, BATT_EFF, BATT_MIN_SOC, BATT_START_SOC = 200.0, 50.0, 0.95, 0.10, 0.50


# --------------------------------------------------------------------------
# 7.1 Synthetic community dataset
# --------------------------------------------------------------------------
def generate_data(days=DAYS, seed=SEED):
    """Reproducible hourly synthetic data for a 120 kWp / ~60 kW-average community."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(START_DATE, periods=days * 24, freq="h")
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    n = len(index)

    temperature = 27 + 7 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 1, n)

    # PV: clear-sky bell 06:00-18:00 x persistent daily cloud factor x hourly flicker
    daily_cloud = np.clip(0.66 + 0.18 * np.sin(np.arange(days) / 5) +
                          rng.normal(0, 0.10, days), 0.2, 0.95)
    cloud = np.repeat(daily_cloud, 24) * np.clip(1 + rng.normal(0, 0.06, n), 0.7, 1.2)
    daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    irradiance = 850 * daylight * cloud
    pv = PV_KWP * (irradiance / 850) * SYSTEM_EFFICIENCY
    pv *= np.maximum(0.75, 1 - TEMP_DERATE * np.maximum(temperature - 25, 0))
    pv = np.clip(pv, 0, None)

    # Load: morning (07:30), midday and evening (20:00) peaks, weekend uplift,
    # cooling term above 28 C, ~5 % random noise
    morning = 13 * np.exp(-0.5 * ((hour - 7.5) / 1.8) ** 2)
    midday = 7 * np.exp(-0.5 * ((hour - 13) / 3.0) ** 2)
    evening = 31 * np.exp(-0.5 * ((hour - 20) / 2.3) ** 2)
    base = 44 + morning + midday + evening
    base = base * np.where(dow >= 5, 1.10, 1.0)
    base = base + np.maximum(temperature - 28, 0) * 1.3
    load = np.clip(base * (1 + rng.normal(0, 0.05, n)), 20, None)

    return pd.DataFrame({
        "pv_kw": pv, "load_kw": load, "temperature_c": temperature,
        "irradiance": irradiance, "hour": hour, "dow": dow,
        # imperfect weather forecast: 12 % noise on irradiance, 1 C on temperature
        "irr_forecast": irradiance * (1 + rng.normal(0, 0.12, n)),
        "temp_forecast": temperature + rng.normal(0, 1, n),
    }, index=index)


# --------------------------------------------------------------------------
# 6.5 Features and 7.2 model comparison
# --------------------------------------------------------------------------
def make_features(df, target):
    """Every target-derived feature uses a lag >= 24 h (no leakage)."""
    x = pd.DataFrame(index=df.index)
    x["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    x["dow"] = df["dow"]
    x["weekend"] = (df["dow"] >= 5).astype(int)
    for lag in (24, 25, 48, 168):
        x[f"{target}_lag{lag}"] = df[target].shift(lag)
    x["roll24_mean"] = df[target].shift(24).rolling(24).mean()
    x["irr_fc"] = df["irr_forecast"]
    x["temp_fc"] = df["temp_forecast"]
    return x.dropna()


def model_factory():
    return {
        "Ridge Regression": lambda: Ridge(alpha=1.0),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED),
    }


def split_chronological(x, y, test_hours=TEST_DAYS * 24):
    return x.iloc[:-test_hours], x.iloc[-test_hours:], y.iloc[:-test_hours], y.iloc[-test_hours:]


def metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    return {"MAE": mae, "RMSE": rmse, "nMAE %": 100 * mae / float(np.mean(actual))}


def evaluate_target(df, target):
    """Train all models on the first 72 days, test on the last 18 days.
    Returns (results dict, actual, predictions dict, fitted models dict, feature columns)."""
    x_all = make_features(df, target)
    y_all = df.loc[x_all.index, target]
    x_tr, x_te, y_tr, y_te = split_chronological(x_all, y_all)

    preds = {"Persistence (t-24h)": df[target].shift(24).loc[y_te.index].to_numpy()}
    fitted = {}
    for name, make in model_factory().items():
        m = make().fit(x_tr, y_tr)
        fitted[name] = m
        preds[name] = np.clip(m.predict(x_te), 0, None)

    results = pd.DataFrame({k: metrics(y_te, v) for k, v in preds.items()}).T
    return results, y_te, {k: pd.Series(v, index=y_te.index) for k, v in preds.items()}, fitted, list(x_te.columns)


# --------------------------------------------------------------------------
# 7.4 Battery dispatch simulation and peak net-load alert
# --------------------------------------------------------------------------
def simulate_dispatch(pv, load, with_battery=True):
    """Rule-based, 1-hour resolution: charge from surplus PV, discharge to cover
    deficit, respecting power limit, 95 % one-way efficiency and 10 % min SoC."""
    soc_kwh = BATT_START_SOC * BATT_KWH
    imports = exports = 0.0
    soc_pct = []
    for p, l in zip(np.asarray(pv, float), np.asarray(load, float)):
        net = l - p
        if with_battery:
            if net < 0:  # surplus -> charge
                charge_in = min(-net, BATT_KW, (BATT_KWH - soc_kwh) / BATT_EFF)
                soc_kwh += charge_in * BATT_EFF
                net += charge_in
            else:        # deficit -> discharge
                avail = max(soc_kwh - BATT_MIN_SOC * BATT_KWH, 0)
                out = min(net, BATT_KW, avail * BATT_EFF)
                soc_kwh -= out / BATT_EFF
                net -= out
        imports += max(net, 0)
        exports += max(-net, 0)
        soc_pct.append(100 * soc_kwh / BATT_KWH)
    total_load = float(np.sum(load))
    return {"import_kwh": imports, "export_kwh": exports,
            "self_sufficiency_pct": 100 * (1 - imports / total_load),
            "soc_pct": pd.Series(soc_pct, index=getattr(pv, "index", None))}


def peak_alert_stats(actual_net, forecast_net, threshold):
    """Precision / recall of the forecast-based alert against actual net load."""
    a = np.asarray(actual_net) > threshold
    f = np.asarray(forecast_net) > threshold
    tp, fp, fn = int(np.sum(a & f)), int(np.sum(~a & f)), int(np.sum(a & ~f))
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": 100 * tp / (tp + fp) if tp + fp else float("nan"),
            "recall": 100 * tp / (tp + fn) if tp + fn else float("nan")}


def dispatch_advice(forecast_net, soc_pct, threshold):
    """Telegram-style message (Report 6.6) for the worst forecast hour."""
    peak_ts = forecast_net.idxmax()
    peak = float(forecast_net.max())
    if peak > threshold:
        return (f"Peak net-load expected at {peak_ts:%H:%M}: forecast {peak:.0f} kW > "
                f"{threshold:.0f} kW threshold; battery SoC {soc_pct:.0f} %. "
                "Hold battery reserve for this hour and shift flexible loads."), True
    return (f"No hour exceeds the {threshold:.0f} kW threshold (max forecast net load "
            f"{peak:.0f} kW at {peak_ts:%H:%M}). Battery can charge from surplus PV."), False


# --------------------------------------------------------------------------
# 6.4 / 8.2 Edge-side logic (Python mirror of the firmware rules, for testing)
# --------------------------------------------------------------------------
NODE_A_FIELDS = ("node", "ts", "pv_v", "pv_i", "pv_w", "lux", "temp", "hum")
RANGES = {"pv_v": (0, 24), "pv_i": (0, 3.2), "temp": (-20, 80)}


def validate_payload(msg):
    """Schema + physical-range check (FR-04). Returns True if acceptable."""
    if not isinstance(msg, dict) or any(k not in msg for k in NODE_A_FIELDS):
        return False
    for key, (lo, hi) in RANGES.items():
        v = msg[key]
        if not isinstance(v, (int, float)) or not (lo < v < hi):
            return False
    return True


def moving_average(values, window=10):
    return float(np.mean(list(values)[-window:]))


def is_stuck(values, window=30):
    """FR-07: 30 consecutive readings with zero variance."""
    v = list(values)[-window:]
    return bool(len(v) == window and max(v) == min(v))


def net_load_alert(net_kw, threshold_kw=40.0):
    return net_kw > threshold_kw


def soc_from_cell_voltage(v, v_min=3.0, v_max=4.2):
    """Linear SoC estimate (6.4), clipped to 0-100 %."""
    return float(np.clip(100 * (v - v_min) / (v_max - v_min), 0, 100))


def low_battery_alert(soc_pct, limit=20.0):
    return soc_pct < limit


def over_temperature_alert(temp_c, limit=55.0):
    return temp_c > limit
