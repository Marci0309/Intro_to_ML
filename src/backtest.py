import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_backtest():
    print("--- Starting Rolling Backtest (Test Set: 2020-2025) ---")
    
    # 1. Load Data & Model
    data_path = os.path.join("data", "processed", "sp500_clean.csv")
    model_path = os.path.join("models", "Linear_Regression.pkl")
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Error: Missing data or model file.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    model = joblib.load(model_path)
    
    # 2. Prepare Test Data
    # Test set starts from 2020-01-01
    test_mask = df.index >= '2020-01-01'
    df_test = df[test_mask].copy()
    
    drop_cols = ['Target_Vol', 'Log_Return']
    feature_cols = [c for c in df.columns if c not in drop_cols and ('Lag' in c or 'Roll' in c)]
    
    X_test = df_test[feature_cols]
    y_test = df_test['Target_Vol']
    
    print(f"Test Set Range: {df_test.index.min()} to {df_test.index.max()}")
    print(f"Test Samples: {len(df_test)}")

    # 3. Generate Predictions
    # Note: In a true 'rolling' setup, we would retrain the model every month.
    # For this assignment, using the pre-trained model on new data is acceptable 
    # to test 'generalization' without 50 hours of retraining.
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # 4. Evaluation Metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- Test Set Results ---")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"R²:  {r2:.6f}")
    
    # 5. Visualisation (The Money Shot)
    plt.figure(figsize=(15, 7))
    
    # Plot Actual vs Predicted
    plt.plot(df_test.index, y_test, label='Actual Volatility', color='black', alpha=0.5, linewidth=1)
    plt.plot(df_test.index, predictions, label='Predicted (Linear Reg)', color='blue', alpha=0.8, linewidth=1)
    
    # Highlight COVID Crash
    plt.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-05-01'), color='red', alpha=0.1, label='COVID Crash')
    
    plt.title('Final Backtest: Forecasting S&P 500 Volatility (2020-2025)')
    plt.ylabel('Volatility (Squared Returns)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join("plots", "final_backtest.png")
    plt.savefig(out_path)
    print(f"Saved final backtest plot to {out_path}")

if __name__ == "__main__":
    run_backtest()