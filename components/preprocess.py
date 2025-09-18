# components/preprocess.py
"""
Preprocessing component for Driving Behavior Analytics (patched version).

Provides two main functions:
- clean_trip_summary(df) -> pd.DataFrame
    Clean and normalize the trip-summary (aggregated) CSV rows. Strips units (%, km, m),
    normalizes column names, coerces types, and returns a cleaned DataFrame ready for mapping.

- clean_telemetry(df, harsh_threshold=2.5, idle_speed_threshold=0.0) -> (pd.DataFrame, dict)
    Clean and normalize second-wise telemetry. Handles sentinel values, parses timestamps,
    coerces numeric columns, computes dt, speed_mps, accel (m/s^2), jerk (m/s^3), marks
    harsh events and idle rows. Also parses ignition_status into 0/1.

Design choices & assumptions:
- Timestamp parsing uses dayfirst=True (matches your samples like '01-08-2025').
- Sentinel values treated as NaN: -32767 and any absolute value >= 1e9 are treated as invalid.
- Distance fields in trip-summary may be in 'm' or 'km' (e.g., '101 m' or '5.30 km').
- Percent fields like '47%' become numeric 47.0.
- IMEI column is dropped (not used in dashboard).
"""

import pandas as pd
import numpy as np
import re
import warnings

_PERCENT_RE = re.compile(r"\s*%\s*")
_UNIT_RE = re.compile(r"([0-9.+\-]+)\s*([a-zA-Z]*)")

SENTINEL_THRESHOLD = 1e9
SENTINEL_LIST = [-32767, 2000000000, -2000000000, 2147483647]


def _strip_percent(s):
    if pd.isna(s):
        return np.nan
    try:
        s = str(s).strip()
        if s.endswith('%'):
            s2 = _PERCENT_RE.sub('', s)
            return float(s2)
        return float(s)
    except Exception:
        return np.nan


def _extract_number_and_unit(s):
    if pd.isna(s):
        return np.nan, ''
    s = str(s).strip()
    if s == '':
        return np.nan, ''
    m = _UNIT_RE.match(s)
    if not m:
        try:
            return float(s), ''
        except Exception:
            return np.nan, ''
    val = m.group(1)
    unit = m.group(2).lower()
    try:
        num = float(val)
    except Exception:
        num = np.nan
    return num, unit


def _safe_numeric(x):
    try:
        if pd.isna(x):
            return np.nan
        nv = float(x)
    except Exception:
        return np.nan
    if int(nv) in SENTINEL_LIST or abs(nv) >= SENTINEL_THRESHOLD:
        return np.nan
    return nv


# ------------------ Trip summary cleaning ------------------
def clean_trip_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Drop imei if present
    if 'imei' in df.columns:
        df.drop(columns=['imei'], inplace=True)

    # Parse timestamp
    ts_col = next((c for c in df.columns if 'start' in c and 'time' in c), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce', dayfirst=True)
        df.rename(columns={ts_col: 'trip_start_time'}, inplace=True)

    # Duration
    dur_candidates = [c for c in df.columns if 'duration' in c]
    if dur_candidates:
        dur_col = dur_candidates[0]
        df[dur_col] = df[dur_col].apply(_safe_numeric)
        df.rename(columns={dur_col: 'trip_duration_mins'}, inplace=True)
    else:
        df['trip_duration_mins'] = np.nan

    # SOC columns
    for col in list(df.columns):
        if 'soc' in col and 'start' in col:
            df[col] = df[col].apply(_strip_percent)
            df.rename(columns={col: 'trip_start_soc_pct'}, inplace=True)
        elif 'soc' in col and 'end' in col:
            df[col] = df[col].apply(_strip_percent)
            df.rename(columns={col: 'trip_end_soc_pct'}, inplace=True)
        elif 'total_soc_consumed' == col:
            df[col] = df[col].apply(_strip_percent)
            df.rename(columns={col: 'total_soc_consumed_pct'}, inplace=True)

    # Distance/range
    distance_cols = [c for c in df.columns if 'distance' in c or c.endswith('_dist') or 'range' in c]
    for col in distance_cols:
        raw = df[col].astype(str)
        parsed = raw.apply(_extract_number_and_unit)
        nums = parsed.apply(lambda t: t[0])
        units = parsed.apply(lambda t: t[1])

        def to_km(val, unit, raw_s):
            if pd.isna(val):
                return np.nan
            if unit == 'm' or (unit == '' and raw_s.strip().endswith('m')):
                return float(val) / 1000.0
            return float(val)

        df[col + '_km'] = [to_km(v, u, rs) for v, u, rs in zip(nums, units, raw)]

    df['trip_duration_mins'] = pd.to_numeric(df['trip_duration_mins'], errors='coerce')

    return df


# ------------------ Telemetry cleaning ------------------
def clean_telemetry(df: pd.DataFrame,
                    harsh_threshold: float = 2.5,
                    idle_speed_threshold: float = 0.0):
    if df is None:
        return df, {}
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Timestamp
    ts_candidates = [c for c in df.columns if 'time' in c or 'timestamp' in c or c == 'time']
    if not ts_candidates:
        warnings.warn('No timestamp found in telemetry')
        return df, {}
    ts_col = ts_candidates[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if ts_col != 'timestamp':
        df.rename(columns={ts_col: 'timestamp'}, inplace=True)

    # Aliases
    alias_map = {
        'throttle_percentage': 'throttle',
        'motor_torque_output': 'torque',
        'mcudc_battery_voltage': 'battery_voltage',
        'ignition_status_from_can': 'ignition_status',
        'output_phase_current': 'output_phase_current',
        'energy_consumed_by_powertrain': 'energy_consumed_by_powertrain'
    }
    df.rename(columns=lambda c: alias_map.get(c, c), inplace=True)

    # Parse ignition to 0/1
    ''' Possible values: 'ON', 'OFF', 1, 0, True, False, '1', '0', etc. '''
    
    if 'ignition_status' in df.columns:
        df['ignition_status'] = (
            df['ignition_status']
            .astype(str).str.upper().str.strip()
            .replace({'ON': '1', 'OFF': '0'})
        )
        df['ignition_status'] = pd.to_numeric(df['ignition_status'], errors='coerce').fillna(0).astype(int)

    # Coerce numerics and drop sentinels
    numeric_cols = [c for c in df.columns if c not in ['timestamp', 'ignition_status']]
    summary = {}
    mask_any = pd.Series(False, index=df.index)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        mask = df[c].isna()
        for v in SENTINEL_LIST:
            mask = mask | (df[c] == v)
        mask = mask | (df[c].abs() >= SENTINEL_THRESHOLD)
        summary[c] = int(mask.sum())
        mask_any = mask_any | mask
    total_drop = int(mask_any.sum())
    summary['total_rows_removed'] = total_drop
    if total_drop > 0:
        df = df.loc[~mask_any].reset_index(drop=True)

    # dt
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(0.0)
    eps = 1e-6
    df['dt_safe'] = df['dt'].replace(0, eps)

    ''' Conerting km/hr to meters/second '''
    if 'speed' in df.columns:
        df['speed_mps'] = df['speed'] * (1000.0 / 3600.0)
    else:
        df['speed_mps'] = 0.0


    










    # accel & jerk
    df['accel'] = df['speed_mps'].diff() / df['dt_safe']
    df['accel'] = df['accel'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-15, 15)
    df['jerk'] = df['accel'].diff() / df['dt_safe']
    df['jerk'] = df['jerk'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-200, 200)

    # harsh_event
    df['harsh_event'] = np.where(df['accel'] > harsh_threshold, 'Hard Accel',
                          np.where(df['accel'] < -harsh_threshold, 'Hard Brake', 'Normal'))

    # idle detection
    if 'throttle' in df.columns:
        df['is_idle'] = np.where((df['speed'] <= idle_speed_threshold) & (df['throttle'] == 0) & (df['ignition_status'] == 1), True, False)
    else:
        df['is_idle'] = np.where((df['speed'] <= idle_speed_threshold) & (df['ignition_status'] == 1), True, False)


    # --- Torque Jerk ---
    if "torque" in df.columns:
        df["torque_diff"] = df["torque"].diff() / df["dt_safe"]
        df["torque_diff"] = df["torque_diff"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Clip extreme spikes
        df["torque_jerk"] = df["torque_diff"].clip(-200, 200)
    else:
        df["torque_jerk"] = 0.0

    # --- Torque Variance (statistical, rolling window) ---
    if "torque" in df.columns:
        window_size = 5   # 5 samples ≈ 5 sec window if 1 Hz data
        df["torque_var"] = df["torque"].rolling(window=window_size, min_periods=2).var()
        df["torque_var"] = df["torque_var"].fillna(0.0)
    else:
        df["torque_var"] = 0.0

    # --- Harsh Events update ---
    df["harsh_event"] = "Normal"

    # Acceleration-based harshness
    df.loc[df["accel"] > harsh_threshold, "harsh_event"] = "Hard Accel"
    df.loc[df["accel"] < -harsh_threshold, "harsh_event"] = "Hard Brake"

    # Overspeed
    overspeed_limit = 50  # km/h threshold
    df.loc[df["speed"] > overspeed_limit, "harsh_event"] = "Overspeed"

    # Torque jerk
    torque_jerk_threshold = 100.0
    df.loc[df["torque_jerk"].abs() > torque_jerk_threshold, "harsh_event"] = "Torque Jerk"

    # Torque variance
    torque_var_threshold = 50.0
    df.loc[df["torque_var"] > torque_var_threshold, "harsh_event"] = "Torque Variance"

    # jerk_threshold = 5.0       # m/s³
    # torque_jerk_threshold = 50.0  # Nm/s

    # # Count jerk events
    # jerk_events = int((group["jerk"].abs() > jerk_threshold).sum()) if "jerk" in group.columns else 0
    # torque_jerk_events = int((group["torque_jerk"].abs() > torque_jerk_threshold).sum()) if "torque_jerk" in group.columns else 0


    df.drop(columns=['dt_safe'], inplace=True)

    return df, summary


if __name__ == '__main__':
    sample = pd.DataFrame({
        'Time': ['01-08-2025 16:40', '01-08-2025 16:43'],
        'speed': [None, '5.38'],
        'throttle_percentage': ['255', '40'],
        'motor_torque_output': ['-32767', '45'],
        'ignition_status_from_can': ['OFF', 'ON']
    })
    cleaned, summary = clean_telemetry(sample)
    print(cleaned)
    print(summary)
