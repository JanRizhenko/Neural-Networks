import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("1. Generating data and plotting ...")
np.random.seed(42)
n_points = 100

data_dict = {
    'Uniform_0_1': np.random.uniform(0, 1, n_points),
    'Uniform_10_100': np.random.uniform(10, 100, n_points),
    'Normal_50_10': np.random.normal(50, 10, n_points),
    'Normal_-5_5': np.random.normal(-5, 5, n_points),
    'Exponential_1': np.random.exponential(1, n_points),
    'Poisson_5': np.random.poisson(5, n_points)
}
df_random = pd.DataFrame(data_dict)

fig, ax = plt.subplots(figsize=(14, 5))
for col in df_random.columns:
    ax.plot(df_random[col], marker='o', linestyle='-', markersize=2, alpha=0.6, label=col)
ax.set_title('1. Generated Random Data (all 6 series)', fontsize=13)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("\n2. Normalizing data (MinMaxScaler) and plotting ...")
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df_random), columns=df_random.columns
)

fig, ax = plt.subplots(figsize=(14, 5))
for col in df_normalized.columns:
    ax.plot(df_normalized[col], marker='s', linestyle='-', markersize=2, alpha=0.6, label=col)
ax.set_title('2. Normalized Data (MinMaxScaler 0–1, all 6 series)', fontsize=13)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



print("\n3. Generating 2 normal distributions with different parameters...")
dist1 = np.random.normal(loc=100, scale=15, size=1000)
dist2 = np.random.normal(loc=500, scale=50, size=1000)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(dist1, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution 1: N(100, 15)')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(dist2, bins=30, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution 2: N(500, 50)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.suptitle('3. Two Generated Normal Distributions', fontsize=14)
plt.tight_layout()
plt.show()



print("\n4. Standardizing distributions (StandardScaler)...")
scaler_std = StandardScaler()
dist1_scaled = scaler_std.fit_transform(dist1.reshape(-1, 1)).flatten()
dist2_scaled = scaler_std.fit_transform(dist2.reshape(-1, 1)).flatten()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(dist1_scaled, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Standardized Distribution 1')
plt.xlabel('Standardized Value (Z-score)')

plt.subplot(1, 2, 2)
plt.hist(dist2_scaled, bins=30, alpha=0.7, color='red', edgecolor='black')
plt.title('Standardized Distribution 2')
plt.xlabel('Standardized Value (Z-score)')
plt.suptitle('4. Standardized Distributions (StandardScaler)', fontsize=14)
plt.tight_layout()
plt.show()

print("\n5 & 6. Loading data from the file 'alarms_no_scalse.csv'...")



try:
    df_alarms = pd.read_csv('alarms_no_scalse.csv')
    print("File loaded successfully. Data shape:", df_alarms.shape)
    print("Column names:", df_alarms.columns.tolist())
    print("First 5 rows:")
    print(df_alarms.head())
except FileNotFoundError:
    print("ERROR: File 'alarms_no_scalse.csv' not found. Please check the file path.")
    exit()




X = df_alarms[['sound', 'distance', 'visibility']].values
y = df_alarms['alarm'].values
print(f"\nX shape: {X.shape}, y shape: {y.shape}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")



print("\n5. Training regression model on ORIGINAL data from the file...")
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model_original = MLPRegressor(
    hidden_layer_sizes=(300, 200, 100),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    batch_size=16,
    learning_rate='adaptive',
    learning_rate_init=0.0005,
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42,
    verbose=False
)

model_original.fit(X_train_o, y_train_o)
y_pred_o = model_original.predict(X_test_o)

mse_original = mean_squared_error(y_test_o, y_pred_o)
r2_original = r2_score(y_test_o, y_pred_o)
print(f"MSE on original data: {mse_original:.6f}")
print(f"R² Score on original data: {r2_original:.4f}")



print("\n6. Normalizing data (MinMaxScaler) and training the model...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_scaled = MLPRegressor(
    hidden_layer_sizes=(300, 200, 100),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    batch_size=16,
    learning_rate='adaptive',
    learning_rate_init=0.0005,
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42,
    verbose=False
)

model_scaled.fit(X_train_s, y_train_s)
y_pred_s = model_scaled.predict(X_test_s)

mse_scaled = mean_squared_error(y_test_s, y_pred_s)
r2_scaled = r2_score(y_test_s, y_pred_s)
print(f"MSE on normalized data: {mse_scaled:.6f}")
print(f"R² Score on normalized data: {r2_scaled:.4f}")



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test_o, y_pred_o, alpha=0.5, edgecolors='k', linewidth=0.3)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(f'Original Data (R² = {r2_original:.4f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_s, y_pred_s, alpha=0.5, edgecolors='k', linewidth=0.3)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(f'Normalized Data (R² = {r2_scaled:.4f})')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Actual vs Predicted: Model Performance Comparison', fontsize=14)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("COMPARISON OF RESULTS (Regression):")
print(f"{'Metric':<15} {'Original Data':<20} {'Normalized Data':<20}")
print("-" * 55)
print(f"{'MSE':<15} {mse_original:<20.6f} {mse_scaled:<20.6f}")
print(f"{'R² Score':<15} {r2_original:<20.4f} {r2_scaled:<20.4f}")
print("="*50)
print("Note: Lower MSE and higher R² indicate better performance.")
print("Normalization helps the neural network converge faster and more reliably.")