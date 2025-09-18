# components/metrics.py
"""
Per-trip KPI / metrics computation for Driving Behavior Analytics.

Functions:
- compute_trip_metrics(telemetry_df, trip_summary_df)
- compute_vehicle_metrics(metrics_df)

Now includes:
- driver_score (0–100)
- classification ("Good", "Average", "Bad")
"""

import pandas as pd
import numpy as np


def compute_trip_metrics(telemetry_df: pd.DataFrame,
                         trip_summary_df: pd.DataFrame,
                         overspeed_limit: float = 80.0,
                         idle_speed_threshold: float = 0.0) -> pd.DataFrame:
    if telemetry_df is None or telemetry_df.empty:
        return pd.DataFrame()

    if 'trip_id' not in telemetry_df.columns:
        raise ValueError("telemetry_df must contain a 'trip_id' column")

    metrics = []

    for trip_id, group in telemetry_df.groupby('trip_id'):
        if pd.isna(trip_id):
            continue
        group = group.sort_values('timestamp')

        # duration
        t_start = group['timestamp'].min()
        t_end = group['timestamp'].max()
        duration_s = (t_end - t_start).total_seconds()
        duration_min = duration_s / 60.0

        # distance from trip summary using trip_start_time
        distance_km = np.nan
        if trip_summary_df is not None and 'trip_start_time' in trip_summary_df.columns and 'total_distance_km' in trip_summary_df.columns:
            match = trip_summary_df.loc[trip_summary_df['trip_start_time'] == t_start, 'total_distance_km']
            if not match.empty:
                distance_km = match.iloc[0]

        # avg / max speed
        avg_speed = group['speed'].mean() if 'speed' in group.columns else np.nan
        max_speed = group['speed'].max() if 'speed' in group.columns else np.nan

        # overspeed %
        if 'speed' in group.columns:
            overspeed_time = group.loc[group['speed'] > overspeed_limit, 'dt'].sum()
            overspeed_pct = (overspeed_time / duration_s * 100.0) if duration_s > 0 else 0.0
        else:
            overspeed_pct = np.nan

        # idle %
        if 'is_idle' in group.columns:
            idle_time = group.loc[group['is_idle'], 'dt'].sum()
            idle_pct = (idle_time / duration_s * 100.0) if duration_s > 0 else 0.0
        else:
            idle_pct = np.nan

        # harsh events
        # if 'harsh_event' in group.columns:
        #     harsh_events = int((group['harsh_event'] != 'Normal').sum())
        # else:
        #     harsh_events = 0

            # harsh events
        hard_accel = int((group['harsh_event'] == 'Hard Accel').sum()) if 'harsh_event' in group.columns else 0
        hard_brake = int((group['harsh_event'] == 'Hard Brake').sum()) if 'harsh_event' in group.columns else 0
        overspeed_events = int((group['speed'] > overspeed_limit).sum()) if 'speed' in group.columns else 0
        jerk_events = int((group['jerk'] > 5).sum()) if 'jerk' in group.columns else 0
        torque_jerk_events = int((group['torque_jerk'] > 50).sum()) if 'torque_jerk' in group.columns else 0

        total_harsh_events = hard_accel + hard_brake + overspeed_events + jerk_events + torque_jerk_events


        # power & energy
        if 'output_phase_current' in group.columns and 'battery_voltage' in group.columns:
            group = group.copy()
            group['power_w'] = group['output_phase_current'] * group['battery_voltage']

            avg_power_w = group['power_w'].mean()
            max_power_w = group['power_w'].max()

            # integrate net, pos, neg (using dt)
            energy_ws_total = (group['power_w'] * group['dt']).sum()
            energy_ws_pos = (group.loc[group['power_w'] > 0, 'power_w'] * 1).sum()
            energy_ws_neg = (group.loc[group['power_w'] < 0, 'power_w'] * 1).sum()

            energy_wh = energy_ws_pos / 3600.0
            energy_kwh = energy_wh / 1000.0

            energy_wh_pos = energy_ws_pos / 3600.0
            energy_wh_neg = abs(energy_ws_neg) / 3600.0
            energy_kwh_pos = energy_wh_pos / 1000.0
            energy_kwh_neg = energy_wh_neg / 1000.0

            regen_pct = (energy_wh_neg / energy_wh_pos * 100.0) if energy_wh_pos > 0 else 0.0
            wh_per_km = np.nan
            pos_power_time = group.loc[group['power_w'] > 0, 'dt'].sum()
            neg_power_time = group.loc[group['power_w'] < 0, 'dt'].sum()
            pos_power_time_pct = (pos_power_time / duration_s * 100.0) if duration_s > 0 else 0.0
            neg_power_time_pct = (neg_power_time / duration_s * 100.0) if duration_s > 0 else 0.0
        else:
            avg_power_w = max_power_w = np.nan
            energy_wh = energy_kwh = np.nan
            energy_wh_pos = energy_wh_neg = np.nan
            energy_kwh_pos = energy_kwh_neg = np.nan
            regen_pct = wh_per_km = np.nan
            pos_power_time_pct = neg_power_time_pct = np.nan

            # --- Torque Jerk & Variance ---
        if "torque_jerk" in group.columns:
            avg_torque_jerk = group["torque_jerk"].mean()
            max_torque_jerk = group["torque_jerk"].abs().max()
        else:
            avg_torque_jerk = np.nan
            max_torque_jerk = np.nan

        if "torque_var" in group.columns:
            avg_torque_var = group["torque_var"].mean()
            max_torque_var = group["torque_var"].max()
        else:
            avg_torque_var = np.nan
            max_torque_var = np.nan

        jerk_threshold = 5.0       # m/s³
        torque_jerk_threshold = 50.0  # Nm/s

        # Count jerk events
        jerk_events = int((group["jerk"].abs() > jerk_threshold).sum()) if "jerk" in group.columns else 0
        torque_jerk_events = int((group["torque_jerk"].abs() > torque_jerk_threshold).sum()) if "torque_jerk" in group.columns else 0


        # --- Driver scoring ---
        driver_score = 100.0
        driver_score -= min(idle_pct, 30) * 0.5
        driver_score -= min(overspeed_pct, 30) * 0.7
        # driver_score -= harsh_events * 1.5
        driver_score -= min(energy_wh, 20) * 2.0
        driver_score = max(driver_score, 0)

        if driver_score >= 75:
            classification = "Good"
        elif driver_score >= 50:
            classification = "Average"
        else:
            classification = "Bad"

        metrics.append({
            'trip_id': trip_id,
            'telemetry_start': t_start,
            'telemetry_end': t_end,
            'duration_s': duration_s,
            'duration_min': duration_min,
            'distance_km': distance_km,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'overspeed_pct': overspeed_pct,
            'idle_pct': idle_pct,
            'Hard Accel': int((group['harsh_event'] == 'Hard Accel').sum()) if 'harsh_event' in group.columns else 0,
            'Hard Brake': int((group['harsh_event'] == 'Hard Brake').sum()) if 'harsh_event' in group.columns else 0,
            'total_harsh_events': total_harsh_events,
            'avg_power_w': avg_power_w,
            'max_power_w': max_power_w,
            'energy_wh': energy_wh,
            'energy_kwh': energy_kwh,
            'energy_wh_pos': energy_wh_pos,
            'energy_wh_neg': energy_wh_neg,
            'energy_kwh_pos': energy_kwh_pos,
            'energy_kwh_neg': energy_kwh_neg,
            'regen_pct': regen_pct,
            'wh_per_km': wh_per_km,
            'pos_power_time_pct': pos_power_time_pct,
            'neg_power_time_pct': neg_power_time_pct,
            'driver_score': driver_score,
            'classification': classification,
            "avg_torque_jerk": avg_torque_jerk,
            "max_torque_jerk": max_torque_jerk,
            "avg_torque_var": avg_torque_var,
            "jerk_events": jerk_events,
            "torque_jerk_events": torque_jerk_events,
            "max_torque_var": max_torque_var
        })

    metrics_df = pd.DataFrame(metrics)

    # recompute efficiency after merge
    if trip_summary_df is not None and not trip_summary_df.empty:
        if 'trip_start_time' in trip_summary_df.columns and 'total_distance_km' in trip_summary_df.columns:
            metrics_df = pd.merge(trip_summary_df[['trip_start_time','total_distance_km']], metrics_df,
                                  left_on='trip_start_time', right_on='telemetry_start', how='right')
            metrics_df['distance_km'] = metrics_df['total_distance_km']
            metrics_df['wh_per_km'] = metrics_df['energy_wh'] / metrics_df['distance_km']
            metrics_df['km_per_kwh'] = metrics_df['distance_km'] / metrics_df['energy_kwh']

    return metrics_df


def compute_vehicle_metrics(metrics_df: pd.DataFrame) -> dict:
    """Aggregate KPIs across all trips."""
    if metrics_df is None or metrics_df.empty:
        return {}

    total_distance = metrics_df["distance_km"].sum(skipna=True)
    total_energy_wh = metrics_df["energy_wh"].sum(skipna=True)
    total_regen_wh = metrics_df["energy_wh_neg"].sum(skipna=True)
    

    overall_eff_wh_per_km = total_energy_wh / total_distance if total_distance > 0 else np.nan
    overall_eff_km_per_kwh = total_distance / (total_energy_wh / 1000.0) if total_energy_wh > 0 else np.nan
    regen_pct_overall = (total_regen_wh / (total_energy_wh + total_regen_wh) * 100.0) if total_energy_wh > 0 else 0.0

    return {
        "Total Distance (km)": total_distance,
        "Total Energy (kWh)": total_energy_wh / 1000.0,
        "Total Regen (kWh)": total_regen_wh / 1000.0,
        "Efficiency (Wh/km)": overall_eff_wh_per_km,
        "Efficiency (km/kWh)": overall_eff_km_per_kwh,
        "Regen % (overall)": regen_pct_overall,
        "Avg Speed (km/h)": metrics_df["avg_speed"].mean(skipna=True),
        "Idle % (avg)": metrics_df["idle_pct"].mean(skipna=True),
        "Overspeed % (avg)": metrics_df["overspeed_pct"].mean(skipna=True),
        "Harsh Accel/Brake Events": metrics_df["harsh_events"].sum(skipna=True),
        # "Harsh Events (total)": metrics_df["total_harsh_events"].sum(skipna=True),
        "Avg Torque Jerk": metrics_df["avg_torque_jerk"].mean(skipna=True),
        "Max Torque Jerk": metrics_df["max_torque_jerk"].max(skipna=True),
        "Avg Torque Variance": metrics_df["avg_torque_var"].mean(skipna=True),
        "Max Torque Variance": metrics_df["max_torque_var"].max(skipna=True),
        "Total Jerk Events": metrics_df["jerk_events"].sum(skipna=True),
        "Total Torque Jerk Events": metrics_df["torque_jerk_events"].sum(skipna=True),
        "Avg Driver Score": metrics_df["driver_score"].mean(skipna=True),
        "All Harsh Events": metrics_df["total_harsh_events"].sum(skipna=True)
    }


if __name__ == "__main__":
    # quick self-test
    tele = pd.DataFrame({
        'timestamp': pd.date_range("2025-09-01 10:00:00", periods=5, freq="s"),
        'speed': [0, 10, 20, 30, 0],
        'dt': [0, 1, 1, 1, 1],
        'is_idle': [True, False, False, False, True],
        'harsh_event': ['Normal','Hard Accel','Normal','Normal','Hard Brake'],
        'trip_id': ['1']*5,
        'output_phase_current': [10, -12, 15, -13, 11],
        'battery_voltage': [50, 50, 50, 50, 50]
    })
    trips = pd.DataFrame({'trip_start_time': [pd.Timestamp("2025-09-01 10:00:00")], 'total_distance_km': [5.3]})
    res = compute_trip_metrics(tele, trips)
    print(res[['trip_id','driver_score','classification']])
