import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from arduino.app_utils import App

def run_ml_training_pipeline():
    print("==================================================")
    print(" Arduino UNO Q SBC - ML Training & Inference")
    print("==================================================")
    
    # 1. Generate Synthetic Sensor Dataset 
    # Features: [Temperature (°C), Humidity (%), Light Level (lux)]
    # Target: Equipment Power Consumption / Cooling Load (Watts)
    np.random.seed(42)
    n_samples = 300
    
    print(f"[INFO] Generating synthetic dataset with {n_samples} samples...")
    temperatures = np.random.uniform(20.0, 45.0, n_samples)
    humidity = np.random.uniform(30.0, 90.0, n_samples)
    light = np.random.uniform(100, 1000, n_samples)
    
    # Simulating a physical target relation with realistic Gaussian noise
    power_load = (1.8 * temperatures) + (0.4 * humidity) + (0.03 * light) + np.random.normal(0, 1.5, n_samples)
    
    X = np.column_stack((temperatures, humidity, light))
    y = power_load
    
    # 2. Split into Training and Testing Sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Training set size : {X_train.shape[0]} samples")
    print(f"[INFO] Testing set size  : {X_test.shape[0]} samples")
    
    # 3. Train the Machine Learning Model
    print("[INFO] Training Random Forest Regressor on Linux MPU...")
    start_time = time.time()
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    training_duration = time.time() - start_time
    print(f"[SUCCESS] Model training completed in {training_duration:.3f} seconds!")
    
    # 4. Evaluate Model Performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n---------------- MODEL EVALUATION ----------------")
    print(f"-> Mean Squared Error (MSE) : {mse:.4f}")
    print(f"-> R-squared (R2) Score     : {r2:.4f}")
    print("--------------------------------------------------")
    
    # 5. Run Live Edge Inference
    print("\n---------------- INFERENCE SAMPLE ----------------")
    # Simulate a new edge data point: Temp = 38.5 C, Humidity = 65.0%, Light = 800 lux
    new_sample = np.array([[38.5, 65.0, 800.0]])
    predicted_load = model.predict(new_sample)
    
    print(f"-> Input Sensor Readings    : Temp={new_sample[0][0]}°C, Hum={new_sample[0][1]}%, Light={new_sample[0][2]} lux")
    print(f"-> Predicted Cooling Load   : {predicted_load[0]:.2f} Watts")
    print("==================================================")

# Execute the training pipeline on application boot
run_ml_training_pipeline()

# Keep the Python container active and running in the background
App.run()
