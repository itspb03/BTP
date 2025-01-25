import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import math

# Load Excel sheets
mpbx_data = pd.read_excel(r"dashboard/MPBX AS ON 13.11.2024.xlsx", sheet_name=None)
pressure_data = pd.read_excel(r"dashboard/PRESSURE CELL AS ON 13.11.2024.xlsx", sheet_name=None)

pd.set_option('future.no_silent_downcasting', True)

# Process Pressure Data
pressure_dfs = []
for name, sheet_df in pressure_data.items():
    sheet_df = sheet_df[['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 8']].drop(0)
    sheet_df['Pressure Develop (Kg/cm^2)'] = -sheet_df['Unnamed: 8']
    pressure_dfs.append(sheet_df)


# Process MPBX Data
mpbx_dfs = []
names = list(mpbx_data.keys())
for i in range(0, len(names), 3):
    sheet1 = mpbx_data[names[i]].drop(0)
    sheet2 = mpbx_data[names[i+1]].drop(0)
    sheet3 = mpbx_data[names[i+2]].drop(0)

    deformation1 = sheet1['Unnamed: 8'] - sheet1['Unnamed: 7']
    deformation2 = sheet2['Unnamed: 8'] - sheet2['Unnamed: 7']
    deformation3 = sheet3['Unnamed: 8'] - sheet3['Unnamed: 7']

    convergence1 = deformation1.diff().fillna(0)
    convergence2 = deformation2.diff().fillna(0)
    convergence3 = deformation3.diff().fillna(0)

    #displacements
    displacement1 = sheet1['Unnamed: 8'] 
    displacement2 = sheet2['Unnamed: 8'] 
    displacement3 = sheet3['Unnamed: 8'] 

    mpbx_dfs.append(pd.DataFrame({
        'DATE': sheet1['Unnamed: 0'],
        'Deformation1': deformation1,
        'Deformation2': deformation2,
        'Deformation3': deformation3,
        'Collar Depth 1': sheet1['Unnamed: 4'],
        'Collar Depth 2': sheet2['Unnamed: 4'],
        'Collar Depth 3': sheet3['Unnamed: 4'],
        'Convergence1': convergence1,
        'Convergence2': convergence2,
        'Convergence3': convergence3,
        'Displacement1' : displacement1,
        'Displacement2' : displacement2,
        'Displacement3' :  displacement3 
    }))

# Streamlit App
st.title("MPBX & Pressure Cell Data Analysis")

# Dropdown for plot selection
plot_type = st.selectbox("Select Plot Type", ["MPBX", "Pressure Cell"])
selected_index = st.selectbox(
    f"Select {plot_type} Plot",
    [f"{plot_type} - {i}" for i in range(len(mpbx_dfs) if plot_type == "MPBX" else len(pressure_dfs))]
)

if plot_type == "MPBX":
    df = mpbx_dfs[int(selected_index.split("-")[1])]
    plot_option = st.radio("Select Plot", ["Convergence vs Date", "Displacement vs Length along MPBX"])
    
    if plot_option == "Convergence vs Date":
        collar_depths = st.multiselect("Select Collar Depths", ["Convergence1", "Convergence2", "Convergence3"],default=["Convergence1", "Convergence2", "Convergence3"])
        fig = go.Figure()
        for col in collar_depths:
            fig.add_trace(go.Scatter(x=df['DATE'], y=df[col], mode='lines+markers', name=col))
        fig.update_layout(
            title={
                'text': "Convergence vs Date",
                'y': 0.95,
                'x' : 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }, 
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
    
        collar_depth_1 = (
            str(subset['Collar Depth 1'].iloc[0])[:3]
            if pd.notna(subset['Collar Depth 1'].iloc[0])
            else 'N/A'
        )
        collar_depth_2 = (
            str(subset['Collar Depth 2'].iloc[0])[:3]
            if pd.notna(subset['Collar Depth 2'].iloc[0])
            else 'N/A'
        )
        collar_depth_3 = (
            str(subset['Collar Depth 3'].iloc[0])[:3]
            if pd.notna(subset['Collar Depth 3'].iloc[0])
            else 'N/A'
        )
    
        fig.add_trace(go.Scatter(
            x=[
                subset['Displacement1'].iloc[0], 
                subset['Displacement2'].iloc[0], 
                subset['Displacement3'].iloc[0]
            ],
            y=[
                subset['Collar Depth 1'].iloc[0], 
                subset['Collar Depth 2'].iloc[0], 
                subset['Collar Depth 3'].iloc[0]
            ],
            mode='lines+markers',
            name=f"Date: {date}",
            line=dict(shape='linear'),
            marker=dict(size=8, symbol='circle', color=colors[i]),
            text=[
                f"Date: {date}<br>Collar Depth: {collar_depth_1}<br>Displacement: {subset['Displacement1'].iloc[0]} mm",
                f"Date: {date}<br>Collar Depth: {collar_depth_2}<br>Displacement: {subset['Displacement2'].iloc[0]} mm",
                f"Date: {date}<br>Collar Depth: {collar_depth_3}<br>Displacement: {subset['Displacement3'].iloc[0]} mm"
            ],
            hovertemplate="%{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title={
            'text': "Displacement vs Length along MPBX",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
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
    df = pressure_dfs[int(selected_index.split("-")[1])]
    fig = go.Figure(go.Scatter(x=df['Unnamed: 0'], y=df['Pressure Develop (Kg/cm^2)'], mode='lines+markers'))
    fig.update_layout(
        title={
            'text': "Pressure Develop Over Time",
            'y': 0.95,  # Adjust vertical position of title
            'x': 0.5,   # Center align title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Pressure (Kg/cm^2)",
        title_font=dict(size=20),  # Adjust font size
    )
    st.plotly_chart(fig)
