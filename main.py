# main.py
import streamlit as st
import pandas as pd
import numpy as np
import traceback
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import threading
import os
import sys
import streamlit.web.cli as stcli

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'False'
os.environ['BROWSER'] = 'default'

def open_browser():
    webbrowser.open_new("http://localhost:8501")

threading.Timer(1, open_browser).start()


log_file = open("app_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

from components import upload, preprocess, mapping, metrics

st.set_page_config(page_title="Driving Behavior Analytics", layout="wide")
st.title("Driving Behavior Analytics Dashboard")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.markdown("## Data Upload")
res = upload.show_upload_panel()
trip_summary_df = res.get("trip_summary_df")
telemetry_df = res.get("telemetry_df")

# Debug toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# ---------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------
if trip_summary_df is not None and telemetry_df is not None:
    try:
        # Clean both files
        trip_summary_df = preprocess.clean_trip_summary(trip_summary_df)
        telemetry_df, cleaning_summary = preprocess.clean_telemetry(telemetry_df)

        # Map trips
        tele_mapped, trips_mapped, mapping_summary = mapping.map_trips_from_summary(
            telemetry_df, trip_summary_df, tolerance_seconds=5
        )
        telemetry_df = tele_mapped
        trip_summary_mapped_df = trips_mapped

        # Compute metrics
        metrics_df = metrics.compute_trip_metrics(telemetry_df, trip_summary_mapped_df)

        # -----------------------------------------------------------------
        # Debug Mode: Show raw/cleaned/mapping details
        # -----------------------------------------------------------------
        if debug_mode:
            st.header(" Debug Data Previews")
            st.subheader("Raw Trip Summary (first 10 rows)")
            st.dataframe(res.get("trip_summary_df").head(10))

            st.subheader("Cleaned Trip Summary")
            st.dataframe(trip_summary_df.head(10))

            st.subheader("Raw Telemetry (first 10 rows)")
            st.dataframe(res.get("telemetry_df").head(10))

            st.subheader("Telemetry Cleaning Summary")
            st.json(cleaning_summary)

            st.subheader("Mapping Summary")
            st.write(mapping_summary)
            st.dataframe(trips_mapped.head(20))

        # -----------------------------------------------------------------
        # 5) Whole-Vehicle KPIs / Summary (Main Section)
        # -----------------------------------------------------------------
        st.header(" Whole-Vehicle KPIs / Summary")

        def metric_card(title, value, color):
            st.markdown(
                f"""
                <div style="
                    background-color:{color};
                    padding:15px;
                    border-radius:12px;
                    text-align:center;
                    margin-bottom:10px;
                    color:white;
                    font-weight:bold;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin:0; font-size:16px;">{title}</h4>
                    <h2 style="margin:0; font-size:24px;">{value}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        try:    
            vehicle_summary = metrics.compute_vehicle_metrics(metrics_df)

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card("Total Distance (km)", f"{vehicle_summary['Total Distance (km)']:.2f}", "#4A90E2")
                metric_card("Total Energy (kWh)", f"{vehicle_summary['Total Energy (kWh)']:.2f}", "#20B2AA")
                metric_card("Total Regen (kWh)", f"{vehicle_summary['Total Regen (kWh)']:.2f}", "#77B300")
                metric_card("Total Jerk Events", f"{vehicle_summary['Jerk Events']}", "#FF6F61")

            with col2:
                metric_card("Efficiency (Wh/km)", f"{vehicle_summary['Efficiency (Wh/km)']:.2f}", "#FFB347")
                metric_card("Trip Efficiency (km/Wh)", f"{vehicle_summary['Efficiency (km/kWh)']:.2f}", "#66BB6A")
                metric_card("Regen % (overall)", f"{vehicle_summary['Regen % (overall)']:.1f}%", "#4DB6AC")
                metric_card("Avg Driver Score", f"{vehicle_summary['Avg Driver Score']:.1f}", "#FF7043")

            with col3:
                metric_card("Avg Speed (km/h)", f"{vehicle_summary['Avg Speed (km/h)']:.2f}", "#5C6BC0")
                metric_card("Idle % (avg)", f"{vehicle_summary['Idle % (avg)']:.1f}%", "#9C27B0")
                metric_card("Overspeed % (avg)", f"{vehicle_summary['Overspeed % (avg)']:.1f}%", "#E57373")
                metric_card("Harsh Events (total)", f"{vehicle_summary['Total Harsh Events']}", "#FF8A65")

        except Exception as e:
            st.error("Failed to compute vehicle summary:")
            st.write(e)
            st.write(traceback.format_exc())

        st.subheader("Harsh Driving Events Summary")
        cols = st.columns(5)
        cols[0].metric("Harsh Accel Events", int(vehicle_summary["Hard Accel Events"]))
        cols[1].metric("Harsh Brake Events", int(vehicle_summary["Hard Brake Events"]))
        cols[2].metric("Jerk Events", int(vehicle_summary["Jerk Events"]))
        cols[3].metric("Torque Jerk Events", int(vehicle_summary["Torque Jerk Events"]))
        cols[4].metric("All Harsh Events", int(vehicle_summary["Total Harsh Events"]))



        # -----------------------------------------------------------------
        # Analytics Tabs (always visible)
        # -----------------------------------------------------------------
        st.header("Analytics")

        tab1, tab2= st.tabs(["Trip-wise Metrics", "Visualizations"])

        with tab1:
            st.subheader("Trip-wise Metrics")
            if not metrics_df.empty:
                # Filter by classification
                categories = ["All", "Good", "Average", "Bad"]
                selected_category = st.selectbox("Filter by Driver Classification", categories)

                df_to_show = metrics_df.copy()
                if selected_category != "All":
                    df_to_show = df_to_show[df_to_show["classification"] == selected_category]

                # Select columns to display
                cols_to_show = [
                "trip_id", "driver_score", "classification",
                "distance_km", "duration_min",
                "avg_speed", "max_speed",
                "overspeed_pct", "idle_pct", "harsh_events","avg_torque_jerk", "max_torque_jerk",
                "avg_torque_var", "max_torque_var",
                "energy_wh", "energy_wh_pos", "energy_wh_neg",
                "wh_per_km", "km_per_wh"
                ]
                available_cols = [c for c in cols_to_show if c in df_to_show.columns]
                df_to_show = df_to_show[available_cols]

            # Apply row coloring
            def highlight_classification(row):
                if row["classification"] == "Good":
                    return ['background-color: #d4edda'] * len(row)   # light green
                elif row["classification"] == "Average":
                    return ['background-color: #fff3cd'] * len(row)   # light yellow
                elif row["classification"] == "Bad":
                    return ['background-color: #f8d7da'] * len(row)   # light red
                else:
                    return [''] * len(row)

            styled_df = df_to_show.style.apply(highlight_classification, axis=1)

            st.dataframe(styled_df, use_container_width=True)


        # with tab1:
        #     st.subheader("Trip-wise Metrics")
        #     if not metrics_df.empty:
        #         # Filter by classification
        #         categories = ["All", "Good", "Average", "Bad"]
        #         selected_category = st.selectbox("Filter by Driver Classification", categories)

        #         df_to_show = metrics_df.copy()
        #         if selected_category != "All":
        #             df_to_show = df_to_show[df_to_show["classification"] == selected_category]

        #         # Add color-coded classification labels
        #         def classify_color(row):
        #             if row["classification"] == "Good":
        #                 return " Good"
        #             elif row["classification"] == "Average":
        #                 return " Average"
        #             elif row["classification"] == "Bad":
        #                 return " Bad"
        #             return "Unknown"

        #         if "classification" in df_to_show.columns:
        #             df_to_show["Driver Classification"] = df_to_show.apply(classify_color, axis=1)

        #         cols_to_show = [
        #             "trip_id", "Driver Classification", "driver_score",
        #             "distance_km", "duration_min",
        #             "avg_speed", "max_speed",
        #             "overspeed_pct", "idle_pct", "harsh_events",
        #             "energy_wh", "energy_wh_pos", "energy_wh_neg",
        #             "wh_per_km", "km_per_kwh"
        #         ]
        #         available_cols = [c for c in cols_to_show if c in df_to_show.columns]
        #         st.dataframe(df_to_show[available_cols].head(50))

        # with tab2:
        #     st.subheader("Whole-Vehicle KPIs / Summary")
        #     try:
        #         vehicle_summary = metrics.compute_vehicle_metrics(metrics_df)
        #         cols = st.columns(3)
        #         for idx, (k, v) in enumerate(vehicle_summary.items()):
        #             with cols[idx % 3]:
        #                 st.metric(label=k, value=f"{v:.2f}" if isinstance(v, (int, float)) else v)
        #     except Exception as e:
        #         st.error("Failed to compute vehicle summary:")
        #         st.write(e)
        #         st.write(traceback.format_exc())

        with tab2:
            st.subheader("Visualizations")
            if not metrics_df.empty:
                trip_ids = metrics_df["trip_id"].dropna().unique()
                selected_trip = st.selectbox("Select a trip for detailed plots", trip_ids)

                if selected_trip:
                    trip_data = telemetry_df[telemetry_df["trip_id"] == selected_trip]

                    st.subheader(f"Trip {selected_trip} — Speed & Power")
                    if "speed" in trip_data.columns:
                        fig_speed = px.line(trip_data, x="timestamp", y="speed", title="Speed vs Time")
                        st.plotly_chart(fig_speed, use_container_width=True)

                    if "power" in trip_data.columns:
                        fig_power = px.line(trip_data, x="timestamp", y="power", title="Power vs Time")
                        st.plotly_chart(fig_power, use_container_width=True)

                st.subheader("Energy Consumption vs Regen (per trip)")
                if "energy_wh_pos" in metrics_df.columns and "energy_wh_neg" in metrics_df.columns:
                    fig_bar = go.Figure()
                    fig_bar.add_bar(x=metrics_df["trip_id"], y=metrics_df["energy_wh_pos"], name="Consumption (Wh)")
                    fig_bar.add_bar(x=metrics_df["trip_id"], y=metrics_df["energy_wh_neg"], name="Regen (Wh)")
                    fig_bar.update_layout(barmode="stack", title="Consumption vs Regen Energy per Trip")
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader(" Trip Score Distribution")
                if "driver_score" in metrics_df.columns:
                    fig_hist = px.histogram(
                        metrics_df, x="driver_score", nbins=10,
                        title="Distribution of Driver Scores",
                        color="classification", barmode="overlay"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # 2. Efficiency Scatter
                st.subheader(" Efficiency Scatter (Distance vs Wh/km)")
                if "wh_per_km" in metrics_df.columns and "distance_km" in metrics_df.columns:
                    fig_scatter = px.scatter(
                        metrics_df, x="distance_km", y="wh_per_km",
                        color="classification", size="driver_score",
                        hover_data=["trip_id"],
                        title="Efficiency per Trip"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # 3. Driver Classification Breakdown
                st.subheader(" Driver Classification Breakdown")
                if "classification" in metrics_df.columns:
                    class_counts = metrics_df["classification"].value_counts().reset_index()
                    class_counts.columns = ["classification", "count"]
                    fig_pie = px.pie(
                        class_counts, values="count", names="classification",
                        title="Proportion of Good / Average / Bad Trips",
                        hole=0.4, color="classification",
                        color_discrete_map={"Good": "#28a745", "Average": "#ffc107", "Bad": "#dc3545"}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.error(" Processing failed")
        st.write(e)
        st.write(traceback.format_exc())
else:
    st.info("Upload both Trip Summary and Telemetry files to get started.")

st.sidebar.markdown("---")
st.sidebar.caption("Toggle Debug Mode to view raw/cleaned data and mapping details.")


if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "main.py", "--server.port=8501", "--server.headless=false"]
    sys.exit(stcli.main())