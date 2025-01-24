import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import plotly.express as px # interactive charts 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from xlsxwriter import Workbook
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import math



# Load your Excel sheets
sh = pd.read_excel(r"file path\MPBX AS ON 03.10.2024.xlsx", sheet_name=None) # change file path with MPBX dataset file path
name = list(sh.keys())

combined_dfs = []

sh1 = pd.read_excel(r"file path\PRESSURE CELL AS ON 03.10.2024.xlsx", sheet_name=None) #  change file path with Pressure cell dataset  file path
name1 = list(sh1.keys())

new_combined_dfs = []




# Process 9 sheets and add 'Pressure Develop (Kg/cm^2)'
for name1, sheet_df in sh1.items():
    # Extract the necessary columns: Date, Collar Depth, Observed Reading
    sheet_df = sheet_df[['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 8']]
    sheet_df= sheet_df.drop(0)


    # # Calculate 'Pressure Develop (Kg/cm^2)' as negative of 'Observed Reading'
    sheet_df['Pressure Develop (Kg/cm^2)'] = -sheet_df['Unnamed: 8']
    combined_df1 = pd.DataFrame({
        'DATE': sheet_df['Unnamed: 0'],
        'Collar Depth': sheet_df['Unnamed: 4'],
        'Pressure Develop (Kg/cm^2)': sheet_df['Pressure Develop (Kg/cm^2)'] 
        
    })
    

    
    new_combined_dfs.append(combined_df1)


pd.set_option('future.no_silent_downcasting', True)

for i in range(0, len(name), 3):
    # Extract the three sheets
    sheet1 = sh[name[i]]
    sheet2 = sh[name[i+1]]
    sheet3 = sh[name[i+2]]

    sheet1 = sheet1.drop(0)
    sheet2 = sheet2.drop(0)
    sheet3 = sheet3.drop(0)

    # Calculate the deformation columns
    deformation1 = sheet1['Unnamed: 8'] - sheet1['Unnamed: 7']
    deformation2 = sheet2['Unnamed: 8'] - sheet2['Unnamed: 7']
    deformation3 = sheet3['Unnamed: 8'] - sheet3['Unnamed: 7']

    #displacements
    displacement1 = sheet1['Unnamed: 8'] 
    displacement2 = sheet2['Unnamed: 8'] 
    displacement3 = sheet3['Unnamed: 8'] 

    # Calculate the convergence columns as the difference between consecutive rows
    convergence1 = deformation1.diff().fillna(0)  # fillna(0) to handle the first row
    convergence2 = deformation2.diff().fillna(0)
    convergence3 = deformation3.diff().fillna(0)

    # Create a new DataFrame with Date, Deformation1, Deformation2, Deformation3, and Collar Depth columns
    combined_df = pd.DataFrame({
        'DATE': sheet1['Unnamed: 0'],
        'Deformation1': deformation1,
        'Deformation2': deformation2,
        'Deformation3': deformation3,
        'Collar Depth 1': sheet1['Unnamed: 4'],
        'Collar Depth 2': sheet2['Unnamed: 4'],
        'Collar Depth 3': sheet3['Unnamed: 4'],
        'Convergence1': convergence1,   # Add convergence for Deformation1
        'Convergence2': convergence2,   # Add convergence for Deformation2
        'Convergence3': convergence3,    # Add convergence for Deformation3
        'Displacement1' : displacement1,
        'Displacement2' : displacement2,
        'Displacement3' :  displacement3 

    })

    # Append the combined DataFrame to the list
    combined_dfs.append(combined_df)







# Assuming combined_dfs (for MPBX) and new_combined_dfs (for Pressure) are already created and populated with data




app = Dash(__name__)

# Extend the dropdown options to include both MPBX and Pressure plots
dropdown_options = [{'label': f'MPBX - {i+1} Plot', 'value': f'MPBX-{i}'} for i in range(len(combined_dfs))]
dropdown_options += [{'label': f'PC - {i} Plot', 'value': f'PC-{i}'} for i in range(len(new_combined_dfs))]

# Add prediction model options (ARIMA, ETS, XGBoost, Random Forest)
model_options = [
    {'label': 'Autoregressive Integrated Moving Average (ARIMA)', 'value': 'arima'},
    {'label': 'Support Vector Machine (SVM)', 'value': 'svm'},
    {'label': 'Long Short-Term Memory (LSTM)', 'value': 'lstm'}
]
   
    
#     {'label': 'XGBoost', 'value': 'xgboost'},
#     {'label': 'Random Forest', 'value': 'rf'}


# Layout of the Dash app
app.layout = html.Div([
    html.H1("MPBX & Pressure Cell Plottings", style={'textAlign': 'center'}),
    
    # Dropdown to select between MPBX and Pressure plots
    html.Label("Select a Plot:"),
    dcc.Dropdown(
        id='plot-selector',
        options=dropdown_options,
        value='MPBX-0',  # default value to show the first MPBX plot
        clearable=False
    ),
 
   
    # Dropdown to select either "Convergence vs Date" or "Deformation vs Collar Depth"
    html.Div(id='plot-type-container', children=[
        html.Label("Select Plot Type:"),
        dcc.Dropdown(
            id='plot-type-selector',
            options=[
                {'label': 'Convergence vs Date', 'value': 'convergence'},
                {'label': 'Collar depth vs Displacement', 'value': 'deformation'}
            ],
            value='convergence',  # default value
            clearable=False
        )
    ], style={'display': 'block'}),  # Initially visible
    
    # This checklist will only apply to MPBX plots and when "Convergence vs Date" is selected
    html.Div(id='collar-options', style={'display': 'none'}, children=[
        html.Label("Select Collar Depth to Display:"),
        dcc.Checklist(
            id='collar-selector',
            options=[
                {'label': 'Collar Depth 1', 'value': 'Convergence1'},
                {'label': 'Collar Depth 2', 'value': 'Convergence2'},
                {'label': 'Collar Depth 3', 'value': 'Convergence3'}
            ],
            value=['Convergence1', 'Convergence2', 'Convergence3'],  # Default to show all plots
            inline=True
        )
    ]),

    # New range slider container for dynamic convergence range filtering
    html.Div(id='convergence-range-container', style={'display': 'none'}, children=[
        html.Label("Set Convergence Range to Hide:"),
        dcc.Input(id='min-convergence', type='number', placeholder="Min", value=-0, style={'margin-right': '10px'}),
        dcc.Input(id='max-convergence', type='number', placeholder="Max", value=0),
        
    ]),

    html.Div(id='dates-selection-container', style={'display': 'none'}, children=[
        html.Label("Select dates:"),
        dcc.DatePickerSingle(
            id='date-picker-1',
            date=combined_dfs[0]['DATE'].iloc[0]
        ),
        dcc.DatePickerSingle(
            id='date-picker-2',
            date=combined_dfs[0]['DATE'].iloc[1]
        ),
        dcc.DatePickerSingle(
            id='date-picker-3',
            date=combined_dfs[0]['DATE'].iloc[2]
        )
    ]),

    # Model selection for PB plots - initially hidden
    html.Div(id='model-container', children=[
        html.Label("Select Prediction Model:"),
        dcc.Dropdown(
            id='model-selector',
            options=model_options,
            value=None,  # No model selected initially
            placeholder="Select a model to see predictions",
            clearable=True
        ),
    ], style={'display': 'none'}),  # Hidden initially

    dcc.Graph(id='graph-output')
])

# Callback to update the visibility of the plot-type dropdown and the graph
@app.callback(
    [Output('graph-output', 'figure'),
     Output('collar-options', 'style'),
     Output('plot-type-container', 'style'),  # Controls visibility of plot-type container
     Output('model-container', 'style'),
     Output('dates-selection-container', 'style'),
     Output('convergence-range-container', 'style')],  # Show range slider for 'Convergence vs Date'  # Controls visibility of model selector for PB plots
    
    [Input('plot-selector', 'value'),
     Input('plot-type-selector', 'value'),
     Input('collar-selector', 'value'),
     Input('model-selector', 'value'),
     Input('date-picker-1', 'date'),
     Input('date-picker-2', 'date'),
     Input('date-picker-3', 'date'), # New input for model selector
     Input('min-convergence', 'value'),
     Input('max-convergence', 'value')]  # New input for range slider
)
def update_graph(selected_plot, plot_type, selected_collar_depths, selected_model,date1, date2, date3, min_convergence, max_convergence):
    

    # Initialize the style for the plot-type container
    plot_type_container_style = {'display': 'block'}
    model_selector_style = {'display': 'none'}  # Hide model selector by default
    convergence_range_container_style = {'display': 'none'}
    dates_selection_container_style = {'display': 'none'}

    
    if selected_plot.startswith('MPBX'):
        # Extract the index for MPBX plots
        selected_plot_idx = int(selected_plot.split('-')[1])
        df = combined_dfs[selected_plot_idx]
        fig = go.Figure()


        # Reset values for model and convergence range when the plot changes
        reset_model_value = None  # Reset model selection to None
        reset_min_convergence = 0  # Default min convergence value
        reset_max_convergence = 0  # Default max convergence value


        # Convert 'DATE' column to datetime
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')  # Handle any parsing errors

        # Ensure 'Pressure Develop (Kg/cm^2)' is numeric
        df['Convergence1'] = pd.to_numeric(df['Convergence1'], errors='coerce')
        df['Convergence2'] = pd.to_numeric(df['Convergence2'], errors='coerce')
        df['Convergence3'] = pd.to_numeric(df['Convergence3'], errors='coerce')
        

        
        # Handle missing or non-numeric data (e.g., forward-fill or drop)
        df = df.dropna(subset=['Convergence1'])  # Drop rows where the target is NaN
        df = df.dropna(subset=['Convergence2'])
        df = df.dropna(subset=['Convergence3'])
          
        

        if plot_type == 'convergence':

            convergence_range_container_style = {'display': 'block'}
            
            max_convergence = max_convergence if max_convergence is not None else float('inf')
            min_convergence = min_convergence if min_convergence is not None else -float('inf')

            # Convergence vs Date plot

            filtered_df_1 = df[
                               ((df['Convergence1'] <= min_convergence) ) | 
                               ((df['Convergence1'] >= max_convergence) )
                              ][['DATE', 'Convergence1']].rename(columns={'DATE': 'Date1', 'Convergence1': 'FilteredConvergence1'}).dropna()

            filtered_df_2 = df[
                               ((df['Convergence2'] <= min_convergence) ) | 
                               ((df['Convergence2'] >= max_convergence) )
                              ][['DATE', 'Convergence2']].rename(columns={'DATE': 'Date2', 'Convergence2': 'FilteredConvergence2'}).dropna()

            filtered_df_3 = df[
                               ((df['Convergence3'] <= min_convergence) ) | 
                               ((df['Convergence3'] >= max_convergence) )
                              ][['DATE', 'Convergence3']].rename(columns={'DATE': 'Date3', 'Convergence3': 'FilteredConvergence3'}).dropna()


            if 'Convergence1' in selected_collar_depths:
                fig.add_trace(go.Scatter(
                    x=filtered_df_1['Date1'], 
                    y=filtered_df_1['FilteredConvergence1'], 
                    mode='lines+markers', 
                    name=f"{df['Collar Depth 1'].iloc[0]}",
                    line=dict(color='blue')
                ))

            if 'Convergence2' in selected_collar_depths:
                fig.add_trace(go.Scatter(
                    x=filtered_df_2['Date2'], 
                    y=filtered_df_2['FilteredConvergence2'], 
                    mode='lines+markers', 
                    name=f"{df['Collar Depth 2'].iloc[0]}",
                    line=dict(color='green')
                ))

            if 'Convergence3' in selected_collar_depths:
                fig.add_trace(go.Scatter(
                    x=filtered_df_3['Date3'], 
                    y=filtered_df_3['FilteredConvergence3'], 
                    mode='lines+markers', 
                    name=f"{df['Collar Depth 3'].iloc[0]}",
                    line=dict(color='red')
                ))

            fig.update_layout(
                title="Convergence (C) vs Date for MPBX",
                xaxis_title=dict(text="Date", font=dict(size=20, weight='bold')),
                yaxis_title=dict(text="Convergence (mm)", font=dict(size=20, weight='bold')),
                hovermode='x unified',
                height=500,
                showlegend=True,
                xaxis=dict(showline=True, linewidth=2, linecolor='black'),
                yaxis=dict(showline=True, linewidth=2, linecolor='black')
            )

           


           

            
            collar_options_style = {'display': 'block'}
            return fig, collar_options_style, plot_type_container_style, model_selector_style, dates_selection_container_style, convergence_range_container_style


        elif plot_type == 'deformation':
            # Deformation vs Collar Depth plot
            # unique_dates = df['DATE'].unique()
            dates_selection_container_style = {'display': 'block'}

            selected_dates = [date1, date2, date3]  # Dates from the date picker inputs
            colors = ['blue', 'green', 'red']  # Define a color for each selected date
            
            

            df['DATE']=df['DATE'].unique()


            for i, date in enumerate(selected_dates):
              # Filter data for the selected date
                date_df = df[df['DATE'] == date]
                
                if (len(date) > 11) :
                    date=date[:-9]
               

                collar_depth_1 = (
                    str(date_df['Collar Depth 1'].iloc[0])[:3] 
                    if isinstance(date_df['Collar Depth 1'].iloc[0], str) 
                    else str(int(date_df['Collar Depth 1'].iloc[0])) 
                    if not math.isnan(date_df['Collar Depth 1'].iloc[0]) 
                    else 'N/A'
                )
                
                collar_depth_2 = (
                    str(date_df['Collar Depth 2'].iloc[0])[:3] 
                    if isinstance(date_df['Collar Depth 2'].iloc[0], str) 
                    else str(int(date_df['Collar Depth 2'].iloc[0])) 
                    if not math.isnan(date_df['Collar Depth 2'].iloc[0]) 
                    else 'N/A'
                )

                collar_depth_3 = (
                    str(date_df['Collar Depth 3'].iloc[0])[:3] 
                    if isinstance(date_df['Collar Depth 3'].iloc[0], str) 
                    else str(int(date_df['Collar Depth 3'].iloc[0])) 
                    if not math.isnan(date_df['Collar Depth 3'].iloc[0]) 
                    else 'N/A'
                )


                fig.add_trace(go.Scatter(
                    x=[date_df['Displacement1'].iloc[0], date_df['Displacement2'].iloc[0], date_df['Displacement3'].iloc[0]],
                    y=[collar_depth_1, collar_depth_2, collar_depth_3],  # Use collar depths for y-axis
                    mode='lines+markers',  # Line graph connecting the points for each date
                    name=f"Date: {date}",
                    line=dict(shape='linear'),
                    marker=dict(size=8, symbol='circle', color=colors[i]),
    
                  # Create a list of hovertexts for each displacement point
                    hovertext=[
                       f"Date: {date}<br>Collar Depth :{collar_depth_1}<br>Displacement: {date_df['Displacement1'].iloc[0]} mm",
                       f"Date: {date}<br>Collar Depth :{collar_depth_2}<br>Displacement: {date_df['Displacement2'].iloc[0]} mm",
                       f"Date: {date}<br>Collar Depth :{collar_depth_3}<br>Displacement: {date_df['Displacement3'].iloc[0]} mm"
                   ],
    
                # Use hovertemplate to display the hovertext for each point
                    hovertemplate="%{hovertext}<extra></extra>"
                ))

            
            fig.update_layout(
                title="Length along MPBX  vs Displacement",
                xaxis_title=dict(text="Displacement (mm)", font=dict(size=20, weight='bold')),
                yaxis_title=dict(text="Length along MPBX (m)", font=dict(size=20, weight='bold')),
                hovermode='x unified',
                height=500,
                showlegend=True,
                xaxis=dict(showline=True, linewidth=2, linecolor='black'),
                yaxis=dict(autorange='reversed', showline=True, linewidth=2, linecolor='black', tickfont=dict(size=10))
            )

        collar_options_style = {'display': 'none'}
        return fig, collar_options_style, plot_type_container_style, model_selector_style, dates_selection_container_style,  {'display': 'none'}

    elif selected_plot.startswith('PC'):
        # Show model selector for PB plots
        
        plot_type_container_style = {'display': 'none'}  # Hide plot-type container for PB

        selected_plot_idx = int(selected_plot.split('-')[1])
        df = new_combined_dfs[selected_plot_idx]
        fig = go.Figure()

        # Plot actual data
        fig.add_trace(go.Scatter(
            x=df['DATE'], 
            y=df['Pressure Develop (Kg/cm^2)'], 
            mode='lines+markers', 
            name=f"Pressure Develop (Kg/cm^2) - P.C {selected_plot_idx}", 
            line=dict(color='purple'),
            marker=dict(size=8, color='purple'),  # Increase marker size if needed
            hovertext=[f"Date: {date}<br>Pressure: {pressure} Kg/cm²" 
                       for date, pressure in zip(df['DATE'], df['Pressure Develop (Kg/cm^2)'])],
            hoverinfo="text",  # Display hover text when hovered
            hovertemplate="<b>Date</b>: %{x}<br><b>Pressure</b>: %{y} Kg/cm²<extra></extra>"
        ))

        fig.update_layout(
           title=f"Pressure Plot for PC - {selected_plot_idx}",
           xaxis_title=dict(text="Date", font=dict(size=20, weight='bold')),
           yaxis_title=dict(text="Pressure Develop (Kg/cm^2)", font=dict(size=20, weight='bold')),
           height=500,
           xaxis=dict(showline=True, linewidth=2, linecolor='black'),
           yaxis=dict(showline=True, linewidth=2, linecolor='black')

        )

        # Convert 'DATE' column to datetime
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')  # Handle any parsing errors

        # Ensure 'Pressure Develop (Kg/cm^2)' is numeric
        df['Pressure Develop (Kg/cm^2)'] = pd.to_numeric(df['Pressure Develop (Kg/cm^2)'], errors='coerce')
        
        # Handle missing or non-numeric data (e.g., forward-fill or drop)
        df = df.dropna(subset=['Pressure Develop (Kg/cm^2)'])  # Drop rows where the target is NaN

        # If a model is selected, show the predicted data
        
        

        return fig, {'display': 'none'}, plot_type_container_style, model_selector_style, {'display': 'none'}, {'display': 'none'}


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)