import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import re

st.set_page_config(layout="wide")
st.title("MPBX & Pressure Cell Data Analysis")

# File upload
st.sidebar.header("Upload Excel Files (Optional)")
uploaded_mpbx_file = st.sidebar.file_uploader("Upload MPBX Excel Sheet", type=["xlsx"])
uploaded_pressure_file = st.sidebar.file_uploader("Upload Pressure Cell Excel Sheet", type=["xlsx"])

# Load Excel Sheets
if uploaded_mpbx_file is not None:
    mpbx_data = pd.read_excel(uploaded_mpbx_file, sheet_name=None)
else:
    mpbx_data = pd.read_excel(r"C:\Users\prana\BTP\MPBX AS ON 13.11.2024.xlsx", sheet_name=None)

if uploaded_pressure_file is not None:
    pressure_data = pd.read_excel(uploaded_pressure_file, sheet_name=None)
else:
    pressure_data = pd.read_excel(r"C:\Users\prana\BTP\PRESSURE CELL AS ON 13.11.2024.xlsx", sheet_name=None)


# Process Pressure Data
pressure_dfs = []
pressure_locations = []

for name, sheet_df in pressure_data.items():
    location_row = sheet_df.iloc[0].astype(str).str.cat(sep=" ")
    match = re.search(r"Location\s*\(?([^)]+)", location_row, re.IGNORECASE)
    location_str = match.group(1).strip() if match else "Unknown"

    sheet_df = sheet_df[['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 8']].drop(0)
    sheet_df['Pressure Develop (Kg/cm^2)'] = -sheet_df['Unnamed: 8']

    pressure_dfs.append(sheet_df)
    pressure_locations.append(location_str)

# Process MPBX Data
mpbx_dfs = []
mpbx_locations = []
names = list(mpbx_data.keys())

for i in range(0, len(names), 3):
    location_row = mpbx_data[names[i]].iloc[0].astype(str).str.cat(sep=" ")
    match = re.search(r"Location\s*\(?([^)]+)", location_row, re.IGNORECASE)
    location_str = match.group(1).strip() if match else "Unknown"

    sheet1 = mpbx_data[names[i]].drop(0)
    sheet2 = mpbx_data[names[i+1]].drop(0)
    sheet3 = mpbx_data[names[i+2]].drop(0)

    deformation1 = sheet1['Unnamed: 8'] - sheet1['Unnamed: 7']
    deformation2 = sheet2['Unnamed: 8'] - sheet2['Unnamed: 7']
    deformation3 = sheet3['Unnamed: 8'] - sheet3['Unnamed: 7']

    convergence1 = deformation1.diff().fillna(0)
    convergence2 = deformation2.diff().fillna(0)
    convergence3 = deformation3.diff().fillna(0)

    displacement1 = sheet1['Unnamed: 8']
    displacement2 = sheet2['Unnamed: 8']
    displacement3 = sheet3['Unnamed: 8']

    df = pd.DataFrame({
        'DATE': sheet1['Unnamed: 0'],
        'Location': sheet1['Unnamed: 2'],
        'Deformation1': deformation1,
        'Deformation2': deformation2,
        'Deformation3': deformation3,
        'Collar Depth 1': sheet1['Unnamed: 4'],
        'Collar Depth 2': sheet2['Unnamed: 4'],
        'Collar Depth 3': sheet3['Unnamed: 4'],
        'Convergence1': convergence1,
        'Convergence2': convergence2,
        'Convergence3': convergence3,
        'Displacement1': displacement1,
        'Displacement2': displacement2,
        'Displacement3': displacement3
    })

    mpbx_dfs.append(df)
    mpbx_locations.append(location_str)


# Streamlit UI
plot_type = st.selectbox("Select Plot Type", ["MPBX", "Pressure Cell"])

if plot_type == "MPBX":
    selected_index = st.selectbox(
        "Select MPBX Plot",
        [f"MPBX - {i}" for i in range(len(mpbx_dfs))]
    )
    index = int(selected_index.split("-")[1])
    df = mpbx_dfs[index]
    location = mpbx_locations[index]

    plot_option = st.radio("Select Plot", ["Convergence vs Date", "Displacement vs Length along MPBX"])

    st.markdown(f"**Location:** {location}")

    if plot_option == "Convergence vs Date":
        collar_depths = {
            "Convergence1": df['Collar Depth 1'].iloc[0],
            "Convergence2": df['Collar Depth 2'].iloc[0],
            "Convergence3": df['Collar Depth 3'].iloc[0]
        }

        selected_curves = st.multiselect(
            "Select Collar Depths",
            list(collar_depths.keys()),
            default=list(collar_depths.keys())
        )

        fig = go.Figure()
        for col in selected_curves:
            depth = collar_depths[col]
            label = f"{col} (Collar Depth: {depth} m)"
            fig.add_trace(go.Scatter(x=df['DATE'], y=df[col], mode='lines+markers', name=label))

        fig.update_layout(
            title=dict(text="Convergence vs Date", y=0.95, x=0.5, xanchor='center', yanchor='top'),
            xaxis_title="Date",
            yaxis_title="Convergence (mm)",
            title_font=dict(size=20),
        )
        st.plotly_chart(fig)

    elif plot_option == "Displacement vs Length along MPBX":
        date1 = st.selectbox("Select First Date", df['DATE'].unique())
        date2 = st.selectbox("Select Second Date", df['DATE'].unique())
        date3 = st.selectbox("Select Third Date", df['DATE'].unique())
        selected_dates = [date1, date2, date3]
        colors = ['blue', 'green', 'red']
        fig = go.Figure()

        for i, date in enumerate(selected_dates):
            subset = df[df['DATE'] == date]
            if subset.empty:
                st.warning(f"No data available for date: {date}")
                continue

            fig.add_trace(go.Scatter(
                x=[subset['Displacement1'].iloc[0], subset['Displacement2'].iloc[0], subset['Displacement3'].iloc[0]],
                y=[subset['Collar Depth 1'].iloc[0], subset['Collar Depth 2'].iloc[0], subset['Collar Depth 3'].iloc[0]],
                mode='lines+markers',
                name=f"Date: {date}",
                line=dict(shape='linear'),
                marker=dict(size=8, symbol='circle', color=colors[i]),
                text=[
                    f"Date: {date}<br>Collar Depth: {subset['Collar Depth 1'].iloc[0]}<br>Displacement: {subset['Displacement1'].iloc[0]} mm",
                    f"Date: {date}<br>Collar Depth: {subset['Collar Depth 2'].iloc[0]}<br>Displacement: {subset['Displacement2'].iloc[0]} mm",
                    f"Date: {date}<br>Collar Depth: {subset['Collar Depth 3'].iloc[0]}<br>Displacement: {subset['Displacement3'].iloc[0]} mm"
                ],
                hovertemplate="%{text}<extra></extra>"
            ))

        fig.update_layout(
            title=dict(text="Displacement vs Length along MPBX", y=0.95, x=0.5, xanchor='center', yanchor='top'),
            xaxis_title=dict(text="Displacement (mm)", font=dict(size=20)),
            yaxis_title=dict(text="Length along MPBX (m)", font=dict(size=20)),
            hovermode='x unified',
            height=500,
            showlegend=True,
            xaxis=dict(showline=True, linewidth=2, linecolor='black'),
            yaxis=dict(autorange='reversed', showline=True, linewidth=2, linecolor='black')
        )
        st.plotly_chart(fig)

elif plot_type == "Pressure Cell":
    selected_index = st.selectbox(
        "Select Pressure Cell Plot",
        [f"Pressure Cell - {i}" for i in range(len(pressure_dfs))]
    )
    index = int(selected_index.split("-")[1])
    df = pressure_dfs[index]
    location = pressure_locations[index]

    st.markdown(f"**Location:** {location}")

    fig = go.Figure(go.Scatter(x=df['Unnamed: 0'], y=df['Pressure Develop (Kg/cm^2)'], mode='lines+markers'))
    fig.update_layout(
        title=dict(text="Pressure Develop Over Time", y=0.95, x=0.5, xanchor='center', yanchor='top'),
        xaxis_title="Date",
        yaxis_title="Pressure (Kg/cm^2)",
        title_font=dict(size=20),
    )
    st.plotly_chart(fig)
