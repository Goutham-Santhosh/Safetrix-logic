import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(0)
num_samples = 1000
data = {
    'param1': np.random.rand(num_samples),
    'param2': np.random.rand(num_samples),
    'param3': np.random.rand(num_samples),
    'param4': np.random.rand(num_samples),
    'param5': np.random.rand(num_samples),
    'param6': np.random.rand(num_samples),
    'param7': np.random.rand(num_samples),
    'param8': np.random.rand(num_samples),
    'param9': np.random.rand(num_samples),
    'flood_occurred': np.random.randint(0, 2, num_samples)  # 0: No, 1: Yes
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9']]
y = df['flood_occurred']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Predict flood probability for new data
new_data = np.array([[0.5, 0.2, 0.1, 0.4, 0.8, 0.7, 0.3, 0.6, 0.5]])  # Example new data
new_data_scaled = scaler.transform(new_data)
flood_probability = model.predict_proba(new_data_scaled)

print(f"Flood Probability: {flood_probability[0][1]:.2f}")
