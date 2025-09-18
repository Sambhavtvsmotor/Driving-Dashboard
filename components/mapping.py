# components/mapping.py
"""
Map trips from trip-summary to telemetry time-series.

Provides:
- map_trips_from_summary(telemetry_df, trip_summary_df, start_col='trip_start_time', duration_col='trip_duration_mins', tolerance_seconds=5)

Behavior:
- Builds time windows for each trip using start time and duration (or next trip start if duration missing).
- Assigns a `trip_id` (string) column to telemetry rows that fall inside any trip window.
- Resolves overlaps by choosing the trip whose start time is closest to the telemetry timestamp.
- Returns: (telemetry_mapped_df, trip_summary_mapped_df, mapping_summary)

Notes:
- This function is intentionally simple and robust (loop-based). It scales well when #trips is much smaller than telemetry rows.
- If you need higher performance for huge datasets we can replace with interval joins.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import warnings


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce', dayfirst=True)


def map_trips_from_summary(
    telemetry_df: pd.DataFrame,
    trip_summary_df: pd.DataFrame,
    start_col: str = 'trip_start_time',
    duration_col: str = 'trip_duration_mins',
    trip_id_col: Optional[str] = None,
    tolerance_seconds: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Map trip windows from trip_summary_df onto telemetry_df.

    Returns:
      telemetry_out: telemetry_df copy with new column 'trip_id' (string or NaN)
      trip_summary_out: trip_summary_df copy with added columns: _window_start, _window_end,
                       telemetry_start, telemetry_end, mapped_rows, mapped_duration_s
      mapping_summary: dictionary with summary stats
    """
    if telemetry_df is None or trip_summary_df is None:
        raise ValueError('Both telemetry_df and trip_summary_df are required')

    # make copies
    tele = telemetry_df.copy().reset_index(drop=True)
    trips = trip_summary_df.copy().reset_index(drop=True)

    # normalize timestamp columns
    if 'timestamp' not in tele.columns:
        # try to find time-like column
        time_cols = [c for c in tele.columns if 'time' in c or 'timestamp' in c]
        if not time_cols:
            raise ValueError('Telemetry has no timestamp column')
        tele.rename(columns={time_cols[0]: 'timestamp'}, inplace=True)
    tele['timestamp'] = _ensure_datetime(tele['timestamp'])
    tele = tele.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    if start_col not in trips.columns:
        # try to find a start time-like column
        cand = next((c for c in trips.columns if 'start' in c and 'time' in c), None)
        if cand is None:
            raise ValueError('Trip summary has no start time column')
        trips.rename(columns={cand: start_col}, inplace=True)
    trips[start_col] = _ensure_datetime(trips[start_col])

    # prepare trip ids
    if trip_id_col is None:
        trips['_trip_id'] = trips.index.astype(str)
    else:
        if trip_id_col not in trips.columns:
            trips['_trip_id'] = trips.index.astype(str)
        else:
            trips['_trip_id'] = trips[trip_id_col].astype(str)

    # compute window start/end
    window_starts = []
    window_ends = []

    for i, row in trips.iterrows():
        s = row[start_col]
        if pd.isna(s):
            # leave NaT window
            start = pd.NaT
            end = pd.NaT
        else:
            start = s - pd.Timedelta(seconds=tolerance_seconds)
            # prefer explicit duration
            if duration_col in trips.columns and not pd.isna(row.get(duration_col)):
                dur_mins = row.get(duration_col)
                try:
                    dur_secs = float(dur_mins) * 60.0
                    end = s + pd.Timedelta(seconds=dur_secs + tolerance_seconds)
                except Exception:
                    end = s + pd.Timedelta(seconds=tolerance_seconds)
            else:
                # infer end as next trip start minus 1 second (will adjust after loop)
                end = pd.NaT
        window_starts.append(start)
        window_ends.append(end)

    trips['_window_start'] = window_starts
    trips['_window_end'] = window_ends

    # fill inferred ends using next trip's start
    for i in range(len(trips)):
        if pd.isna(trips.loc[i, '_window_end']):
            if i + 1 < len(trips) and not pd.isna(trips.loc[i + 1, '_window_start']):
                inferred_end = trips.loc[i + 1, '_window_start'] - pd.Timedelta(seconds=1)
                trips.loc[i, '_window_end'] = inferred_end
            else:
                # as fallback set end to start + 1 hour
                if not pd.isna(trips.loc[i, '_window_start']):
                    trips.loc[i, '_window_end'] = trips.loc[i, '_window_start'] + pd.Timedelta(hours=1)

    # initialize trip assignment column on telemetry
    tele['_trip_id'] = pd.NA

    # vector of assigned trip starts per telemetry row (for overlap resolution)
    assigned_trip_start = [pd.NaT] * len(tele)

    # loop over trips and assign
    for i, trow in trips.iterrows():
        trip_id = trow['_trip_id']
        wstart = trow['_window_start']
        wend = trow['_window_end']
        if pd.isna(wstart) or pd.isna(wend):
            continue
        # boolean mask where timestamp is within window
        mask = (tele['timestamp'] >= wstart) & (tele['timestamp'] <= wend)
        if not mask.any():
            continue

        # For rows currently unassigned -> assign directly
        to_assign_idxs = tele.loc[mask & tele['_trip_id'].isna()].index
        tele.loc[to_assign_idxs, '_trip_id'] = trip_id
        for idx in to_assign_idxs:
            assigned_trip_start[idx] = wstart

        # For rows already assigned (overlap) -> resolve by nearest trip start
        overlap_idxs = tele.loc[mask & tele['_trip_id'].notna()].index
        for idx in overlap_idxs:
            current_start = assigned_trip_start[idx]
            # compute which start is closer to telemetry timestamp
            ts = tele.at[idx, 'timestamp']
            dist_current = abs((ts - current_start).total_seconds()) if not pd.isna(current_start) else np.inf
            dist_new = abs((ts - wstart).total_seconds()) if not pd.isna(wstart) else np.inf
            if dist_new < dist_current:
                # replace assignment
                tele.at[idx, '_trip_id'] = trip_id
                assigned_trip_start[idx] = wstart

    # convert _trip_id to trip_id column (NaN where not assigned)
    tele['trip_id'] = tele['_trip_id'].where(tele['_trip_id'].notna(), pd.NA)
    tele.drop(columns=['_trip_id'], inplace=True)

    # compute per-trip mapping metadata
    mapped_rows = []
    telemetry_start = []
    telemetry_end = []
    mapped_durations_s = []

    for i, trow in trips.iterrows():
        tid = trow['_trip_id']
        if pd.isna(tid):
            mapped_rows.append(0)
            telemetry_start.append(pd.NaT)
            telemetry_end.append(pd.NaT)
            mapped_durations_s.append(0)
            continue
        mask = tele['trip_id'] == tid
        count = int(mask.sum())
        mapped_rows.append(count)
        if count == 0:
            telemetry_start.append(pd.NaT)
            telemetry_end.append(pd.NaT)
            mapped_durations_s.append(0)
        else:
            ts_start = tele.loc[mask, 'timestamp'].min()
            ts_end = tele.loc[mask, 'timestamp'].max()
            telemetry_start.append(ts_start)
            telemetry_end.append(ts_end)
            mapped_durations_s.append((ts_end - ts_start).total_seconds())

    trips['telemetry_start'] = telemetry_start
    trips['telemetry_end'] = telemetry_end
    trips['mapped_rows'] = mapped_rows
    trips['mapped_duration_s'] = mapped_durations_s

    # mapping summary
    mapping_summary = {
        'num_trips': len(trips),
        'num_trips_mapped': int(sum([1 for v in mapped_rows if v > 0])),
        'total_mapped_rows': int(tele['trip_id'].notna().sum()),
        'total_telemetry_rows': int(len(tele)),
        'unmapped_telemetry_rows': int(tele['trip_id'].isna().sum())
    }

    return tele, trips, mapping_summary


# quick test when run as script
if __name__ == '__main__':
    import pandas as _pd
    tele = _pd.DataFrame({
        'timestamp': ['01-09-2025 10:18:00', '01-09-2025 10:18:05', '01-09-2025 10:20:00'],
        'speed': [0, 5, 10]
    })
    trips = _pd.DataFrame({
        'trip_start_time': ['01-09-2025 10:18:00', '01-09-2025 10:19:50'],
        'trip_duration_mins': [2.0, 1.0]
    })
    tele_df, trips_df, summary = map_trips_from_summary(tele, trips, tolerance_seconds=5)
    print(tele_df)
    print(trips_df)
    print(summary)
