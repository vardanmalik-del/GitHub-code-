"""Software tests mirroring report Section 8.2 (UT-01..UT-08, ST-01..ST-07)."""
import numpy as np
import pytest
import microgrid_core as c


@pytest.fixture(scope="module")
def exp():
    df = c.generate_data()
    return df, c.evaluate_target(df, "pv_kw"), c.evaluate_target(df, "load_kw")


GOOD = {"node": "A", "ts": 1793600000, "pv_v": 5.82, "pv_i": 0.91, "pv_w": 5.30,
        "lux": 42150, "temp": 33.2, "hum": 41.5}


def test_ut01_valid_payload():
    assert c.validate_payload(GOOD) is True

def test_ut02_out_of_range_voltage():
    assert c.validate_payload({**GOOD, "pv_v": 48}) is False

def test_ut03_missing_field():
    bad = dict(GOOD); bad.pop("lux")
    assert c.validate_payload(bad) is False

def test_ut04_moving_average():
    assert c.moving_average(range(1, 11)) == 5.5

def test_ut05_stuck_sensor_flagged():
    assert c.is_stuck([3.3] * 30) is True

def test_ut06_noisy_window_not_stuck():
    rng = np.random.default_rng(0)
    assert c.is_stuck(3.3 + rng.normal(0, 0.05, 30)) is False

def test_ut07_net_load_alert():
    assert c.net_load_alert(45) is True and c.net_load_alert(35) is False

def test_ut08_soc_from_cell_voltage():
    assert c.soc_from_cell_voltage(3.60) == pytest.approx(50.0)

def test_st01_dataset_integrity():
    df = c.generate_data()
    assert len(df) == 2160 and int(df.isna().sum().sum()) == 0

def test_st02_pv_forecast_non_negative(exp):
    assert exp[1][2]["Ridge Regression"].min() >= 0

def test_st03_pv_night_near_zero(exp):
    pred = exp[1][2]["Ridge Regression"]
    night = pred[(pred.index.hour >= 20) | (pred.index.hour <= 4)]
    assert night.mean() < 0.5

def test_st04_ridge_beats_persistence_pv(exp):
    r = exp[1][0]["MAE"]
    assert r["Ridge Regression"] < r["Persistence (t-24h)"]

def test_st05_ridge_beats_persistence_load(exp):
    r = exp[2][0]["MAE"]
    assert r["Ridge Regression"] < r["Persistence (t-24h)"]

def test_st06_load_nmae_below_8(exp):
    assert exp[2][0].loc["Ridge Regression", "nMAE %"] < 8

def test_st07_peak_alert_targets(exp):
    _, pv, ld = exp
    s = c.peak_alert_stats(ld[1] - pv[1],
                           ld[2]["Ridge Regression"] - pv[2]["Ridge Regression"], 40)
    assert s["recall"] >= 85 and s["precision"] >= 90
