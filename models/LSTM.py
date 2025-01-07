import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates




sh1 = pd.read_excel(r"C:\Users\prana\BTP\PRESSURE CELL AS ON 13.11.2024.xlsx", sheet_name=None)
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


# Access the 7th sheet

from sklearn.preprocessing import MinMaxScaler

# Extract the 7th sheet and clean it
seventh_sheet = new_combined_dfs[7]
seventh_sheet['DATE'] = pd.to_datetime(seventh_sheet['DATE'], errors='coerce')
seventh_sheet = seventh_sheet.dropna().sort_values(by='DATE')
seventh_sheet.set_index('DATE', inplace=True)

# Ensure 'Pressure Develop (Kg/cm^2)' is numeric
seventh_sheet['Pressure Develop (Kg/cm^2)'] = pd.to_numeric(seventh_sheet['Pressure Develop (Kg/cm^2)'], errors='coerce')
seventh_sheet = seventh_sheet.dropna()

# Extract the pressure data
pressure_data = seventh_sheet['Pressure Develop (Kg/cm^2)'].values.reshape(-1, 1)

# Normalize the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(pressure_data)

# Define training size
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for LSTM (X: input sequence, y: target value)
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10  # Number of time steps to look back
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)



# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predict on test data
predicted_values = model.predict(X_test)

# Rescale the predictions and actual values
predicted_values = scaler.inverse_transform(predicted_values)
y_test_rescaled = scaler.inverse_transform(y_test)

# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_rescaled, predicted_values)
r2 = r2_score(y_test_rescaled, predicted_values)




# Plot actual vs predicted values
plt.figure(figsize=(16, 8))
plt.plot(seventh_sheet.index[-len(y_test_rescaled):], y_test_rescaled, label='Actual Data', color='blue')
plt.plot(seventh_sheet.index[-len(predicted_values):], predicted_values, label='Predicted Data (LSTM)', color='red')

# Add labels and legend
# Add title, bold x and y labels
plt.title('Pressure Develop vs Date (LSTM)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel('Pressure Develop (Kg/cm^2)', fontsize=14, fontweight='bold')


# Customize x-axis ticks
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Set tick interval (e.g., every 2 weeks)

# Emphasize axis lines
plt.gca().spines['bottom'].set_linewidth(2)  # X-axis line
plt.gca().spines['left'].set_linewidth(2)    # Y-axis line

# Customize ticks
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)



# Add a grid and legend
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)





# Add MSE and Accuracy as text annotations
text_x = seventh_sheet.index[-len(predicted_values) // 2]  # Midpoint for annotation
plt.text(
    text_x, 
    max(y_test_rescaled.max(), predicted_values.max()) * 0.95,  # Slightly below the max value
    f'MSE: {mse:.4f}\nR²: {r2:.2f}%', 
    fontsize=12, 
    color='green',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Show plot
plt.show()












