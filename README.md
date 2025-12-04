# Introduction to Machine Learning – Assignment 3 Proposal  
**Xiaohan Bao (S4036298), Marcell Szőkedencsi (s5173191), Ana Abadias Alfonso (s5460824)**  
**December 4, 2025**

---

## Motivation & General Idea
Financial market volatility is a nonlinear and challenging time-series prediction problem. Given our interest in financial markets and existing background in finance, the aim of this project is to build and evaluate machine learning models to forecast short-term volatility of the S&P 500 index.  
This problem is sufficiently challenging for an ML assignment and aligns well with course techniques and our personal interests.  
:contentReference[oaicite:1]{index=1}

---

## Dataset
We will use historical price data of the S&P 500 index (**ticker: ^GSPC**) retrieved using Python’s **yfinance** library.  
The dataset spans from the earliest available observations (≈1927) up to ~December 2025.

Included variables:
- Daily returns  
- Realized volatility (RV) as squared daily returns  
- Monthly returns (for extended analysis)  
- Lag features  
- Rolling-window features (rolling mean, rolling volatility)  
:contentReference[oaicite:2]{index=2}

---

## Task Definition
Our goal is to develop a forecasting model for **next-day volatility** of the S&P 500 index.

The project workflow:
1. Preprocess historical price data  
2. Construct lagged and rolling-window features  
3. Train various ML models  

We will use:
- **Grid search** for systematic hyperparameter tuning  
- **Rolling-window evaluation** to mimic real-time forecasting and avoid look-ahead bias  
- **Time-series train/validation/test splits**  
- Standard regression metrics: **MSE**, **MAE**, **R²**  

Backtesting will be applied to evaluate generalization on unseen future periods and compare how models behave under different market conditions.  
:contentReference[oaicite:3]{index=3}

---

## Models to Be Trained
Volatility forecasting often begins with econometric models, but in this course we focus on **machine learning approaches** and compare them.  
We also found a relevant working paper that we plan to reference and partially implement.  
:contentReference[oaicite:4]{index=4}

### Classical ML Models (Scikit-Learn)
- Linear Regression  
- Ridge Regression (L1/L2 regularization)  
- Support Vector Regression (SVR)  
- Random Forest Regressor  
- Gradient Boosted Trees (XGBoost or LightGBM)

### Neural Network Models (PyTorch)
- Multilayer Perceptron (MLP)  
- LSTM for time-series forecasting  
:contentReference[oaicite:5]{index=5}

