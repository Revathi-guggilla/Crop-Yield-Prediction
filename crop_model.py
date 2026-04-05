
# CROP YIELD PREDICTION PROJECT

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset

data = pd.read_csv("crop_dataset.csv")

print("Dataset Loaded Successfully ✅")
print(data.head())

# 3. Preprocessing

# Handle missing values
data = data.dropna()

# Drop unnecessary columns
data = data.drop("Unnamed: 0", axis=1)

# Encode categorical columns using one-hot encoding for better performance
data = pd.get_dummies(data, columns=['Area', 'Item'], drop_first=True)

# Separate features & target
X = data.drop("hg/ha_yield", axis=1)
y = data["hg/ha_yield"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. EDA (Visualization)

plt.figure()
sns.scatterplot(x=data['average_rain_fall_mm_per_year'], y=data['hg/ha_yield'])
plt.title("Rainfall vs Yield")
plt.savefig("rainfall_vs_yield.png")
# plt.show()  # Removed to avoid blocking in terminal

plt.figure()
sns.scatterplot(x=data['avg_temp'], y=data['hg/ha_yield'])
plt.title("Temperature vs Yield")
plt.savefig("temp_vs_yield.png")
# plt.show()  # Removed to avoid blocking in terminal

# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Model Training

model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

print("Model Trained ✅")

# 7. Prediction

y_pred = model.predict(X_test)

# 8. Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# 9. Visualization (Actual vs Predicted)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.savefig("actual_vs_predicted.png")
# plt.show()  # Removed to avoid blocking in terminal

# 10. Farmer Recommendation

print("\nFarmer Recommendations:")

avg_rainfall = data['average_rain_fall_mm_per_year'].mean()
avg_temp = data['avg_temp'].mean()

if avg_rainfall > 100:
    print("Suitable crops: Rice, Sugarcane")
elif avg_rainfall > 50:
    print("Suitable crops: Wheat, Maize")
else:
    print("Suitable crops: Millets, Pulses")

if avg_temp > 30:
    print("High temperature crops: Cotton, Groundnut")
else:
    print("Moderate temperature crops: Wheat, Barley")