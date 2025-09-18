# components/upload.py
"""
Upload component for Driving Behavior Analytics Dashboard.

Provides a Streamlit sidebar UI to upload trip-summary and telemetry files
(CSV/XLSX), or alternatively specify local file paths. Returns loaded
DataFrames along with their sources.

This patched version:
- Skips 'sep=' row if present in CSV files.
- Drops 'Unnamed' columns automatically.
"""

import streamlit as st
import pandas as pd


@st.cache_data(ttl=3)
def read_file(file_or_path, kind: str = "csv") -> pd.DataFrame:
    """Read CSV or Excel into pandas DataFrame, with cleaning for sep=/Unnamed columns."""
    if file_or_path is None:
        return None
    try:
        if kind == "csv":
            # Check first line for "sep="
            skip_first = False
            try:
                if hasattr(file_or_path, "read"):  # file_uploader gives a buffer
                    first_line = file_or_path.readline().decode("utf-8", errors="ignore")
                    file_or_path.seek(0)
                else:  # local path
                    with open(file_or_path, "r", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline()
                if "sep=" in first_line.lower():
                    skip_first = True
            except Exception:
                pass

            df = pd.read_csv(file_or_path, skiprows=1 if skip_first else 0)

        elif kind in ["xlsx", "xls"]:
            df = pd.read_excel(file_or_path)
        else:
            return None

        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        return df

    except Exception as e:
        st.error(f"Failed to read {kind} file: {e}")
        return None


def show_upload_panel():
    """
    Show sidebar upload panel and return dictionary:
    {
        "trip_summary_df": DataFrame or None,
        "telemetry_df": DataFrame or None,
        "trip_summary_source": str,
        "telemetry_source": str
    }
    """
    st.sidebar.markdown("### Upload Data Files")

    trip_summary_df = None
    telemetry_df = None
    trip_summary_source = None
    telemetry_source = None

    # --- Trip Summary Upload ---
    st.sidebar.subheader("Trip Summary (aggregated)")
    trip_file = st.sidebar.file_uploader("Upload Trip Summary CSV/XLSX", type=["csv", "xlsx"], key="trip_summary")
    trip_local = st.sidebar.text_input("Or provide local path to Trip Summary file")
    if trip_file is not None:
        ext = trip_file.name.split(".")[-1].lower()
        trip_summary_df = read_file(trip_file, "csv" if ext == "csv" else "xlsx")
        trip_summary_source = f"uploaded:{trip_file.name}"
    elif trip_local:
        ext = trip_local.split(".")[-1].lower()
        trip_summary_df = read_file(trip_local, "csv" if ext == "csv" else "xlsx")
        trip_summary_source = f"path:{trip_local}"

    # --- Telemetry Upload ---
    st.sidebar.subheader("Telemetry (time-series)")
    telem_file = st.sidebar.file_uploader("Upload Telemetry CSV/XLSX", type=["csv", "xlsx"], key="telemetry")
    telem_local = st.sidebar.text_input("Or provide local path to Telemetry file")
    if telem_file is not None:
        ext = telem_file.name.split(".")[-1].lower()
        telemetry_df = read_file(telem_file, "csv" if ext == "csv" else "xlsx")
        telemetry_source = f"uploaded:{telem_file.name}"
    elif telem_local:
        ext = telem_local.split(".")[-1].lower()
        telemetry_df = read_file(telem_local, "csv" if ext == "csv" else "xlsx")
        telemetry_source = f"path:{telem_local}"

    return {
        "trip_summary_df": trip_summary_df,
        "telemetry_df": telemetry_df,
        "trip_summary_source": trip_summary_source,
        "telemetry_source": telemetry_source,
    }


if __name__ == "__main__":
    st.title("Debug: Upload Component")
    res = show_upload_panel()
    st.write("Trip summary source:", res["trip_summary_source"])
    st.write("Telemetry source:", res["telemetry_source"])
    if res["trip_summary_df"] is not None:
        st.write("Trip summary head:")
        st.write(res["trip_summary_df"].head())
    if res["telemetry_df"] is not None:
        st.write("Telemetry head:")
        st.write(res["telemetry_df"].head())
