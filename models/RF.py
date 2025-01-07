import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

# Load data from Excel
sh1 = pd.read_excel(r"C:\Users\prana\BTP\PRESSURE CELL AS ON 13.11.2024.xlsx", sheet_name=None)
name1 = list(sh1.keys())

new_combined_dfs = []

# Process 9 sheets and add 'Pressure Develop (Kg/cm^2)'
for name1, sheet_df in sh1.items():
    # Extract the necessary columns: Date, Collar Depth, Observed Reading
    sheet_df = sheet_df[['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 8']]
    sheet_df = sheet_df.drop(0)

    # Calculate 'Pressure Develop (Kg/cm^2)' as negative of 'Observed Reading'
    sheet_df['Pressure Develop (Kg/cm^2)'] = -sheet_df['Unnamed: 8']
    combined_df1 = pd.DataFrame({
        'DATE': sheet_df['Unnamed: 0'],
        'Collar Depth': sheet_df['Unnamed: 4'],
        'Pressure Develop (Kg/cm^2)': sheet_df['Pressure Develop (Kg/cm^2)']
    })

    new_combined_dfs.append(combined_df1)

# Access the 7th sheet
seventh_sheet = new_combined_dfs[7]
seventh_sheet['DATE'] = pd.to_datetime(seventh_sheet['DATE'], errors='coerce')
seventh_sheet = seventh_sheet.dropna().sort_values(by='DATE')
seventh_sheet.set_index('DATE', inplace=True)

# Ensure 'Pressure Develop (Kg/cm^2)' is numeric
seventh_sheet['Pressure Develop (Kg/cm^2)'] = pd.to_numeric(seventh_sheet['Pressure Develop (Kg/cm^2)'], errors='coerce')
seventh_sheet = seventh_sheet.dropna()

# Extract the pressure data
pressure_data = seventh_sheet['Pressure Develop (Kg/cm^2)'].values.reshape(-1, 1)

# Normalize the data for Random Forest
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(pressure_data)

# Define training size
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for training
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10  # Number of time steps to look back
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape the data to fit the RandomForestRegressor input format
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# Reshape y_train and y_test to be 1D arrays using ravel()
y_train = y_train.ravel()
y_test = y_test.ravel()


# Simulated Annealing for hyperparameter optimization
def simulated_annealing(X_train, y_train, X_test, y_test, max_iter=1000, initial_temp=1000, cooling_rate=0.99):
    # Initial Random Forest hyperparameters
    best_n_estimators = 10
    best_max_depth = 5
    best_mse = float('inf')
    best_r2 = -float('inf')

    current_n_estimators = best_n_estimators
    current_max_depth = best_max_depth
    current_mse = best_mse
    current_r2 = best_r2

    temperature = initial_temp

    for iteration in range(max_iter):
        # Randomly select new hyperparameters
        new_n_estimators = int(np.random.uniform(10, 200))
        new_max_depth = int(np.random.uniform(1, 20))

        # Train the RandomForest model with these hyperparameters
        rf_model = RandomForestRegressor(n_estimators=new_n_estimators, max_depth=new_max_depth, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # If the new solution is better or accepted by simulated annealing, update best solution
        if mse < best_mse or np.random.rand() < np.exp((current_mse - mse) / temperature):
            best_mse = mse
            best_r2 = r2
            best_n_estimators = new_n_estimators
            best_max_depth = new_max_depth

        # Cooling the temperature
        temperature *= cooling_rate

    return best_n_estimators, best_max_depth, best_mse, best_r2

# Run Simulated Annealing to find the best hyperparameters
best_n_estimators, best_max_depth, best_mse, best_r2 = simulated_annealing(X_train, y_train, X_test, y_test)

# Print the results of optimization
print(f"Best n_estimators: {best_n_estimators}")
print(f"Best max_depth: {best_max_depth}")
print(f"Best MSE: {best_mse}")
print(f"Best R²: {best_r2}")

# Train a Random Forest model with the best hyperparameters found
rf_model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Calculate the MSE and R² on the test set
mse_rf = mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)

# Plot actual vs predicted values
plt.figure(figsize=(16, 8))
plt.plot(seventh_sheet.index[-len(y_test):], y_test, label='Actual Data', color='blue')
plt.plot(seventh_sheet.index[-len(y_pred):], y_pred, label='Predicted Data (RF)', color='red')

# Add labels and legend
plt.title('Pressure Develop vs Date (Random Forest)', fontsize=16, fontweight='bold')
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

# Add grid and legend
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)

# Add MSE and R² as text annotations
text_x = seventh_sheet.index[-len(y_pred) // 2]  # Midpoint for annotation
plt.text(
    text_x, 
    max(y_test.max(), y_pred.max()) * 0.95,  # Slightly below the max value
    f'MSE: {mse_rf:.4f}\nR²: {r2_rf:.2f}', 
    fontsize=12, 
    color='green',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Show plot
plt.show()

