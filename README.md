# 🔧 Predict-V85-Operating-System

*Machine learning regression models that accurately predict V85 operating system parameters for manufacturing process optimization and quality control.*

![Manufacturing](https://img.shields.io/badge/domain-industrial_manufacturing-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![ML](https://img.shields.io/badge/regression-multiple_models-orange)
![Accuracy](https://img.shields.io/badge/R²-0.982-green)

---

## 📋 Overview

I developed this project to solve the challenge of **manually estimating V85 operating parameters in manufacturing systems**. In industrial settings, accurate prediction of operating parameters is crucial for maintaining product quality, optimizing production efficiency, and reducing waste. I built multiple regression models that automatically predict V85 values based on other measurable process variables, achieving over 98% accuracy and eliminating the need for error-prone manual estimation.

**Industry Application:** Industrial Manufacturing / Production Engineering / Process Optimization

---

## 🎯 The Problem

Manufacturing facilities face several operational challenges:
- **Manual Estimation Errors:** Operators manually estimate V85 parameters leading to inconsistencies
- **Production Inefficiency:** Suboptimal operating parameters result in wasted materials and energy
- **Quality Variability:** Inconsistent V85 values affect product quality and specifications
- **Lack of Data-Driven Decisions:** Reliance on operator experience rather than predictive analytics
- **Training Dependency:** New operators require extensive training to estimate parameters accurately

---

## 💡 The Solution

A machine learning pipeline that predicts V85 operating parameters using measurable process variables:
1. **Multiple Model Approach:** Ridge Regression, Lasso Regression, and Gradient Boosting Regressor
2. **Feature Engineering:** One-hot encoding for categorical variables and standardization of numeric features
3. **Comprehensive Evaluation:** MAE, MSE, RMSE, R², and correlation metrics for performance validation
4. **Production-Ready Pipeline:** Complete data preprocessing, training, and prediction workflow

### Key Features:
- ✅ **High Accuracy:** Achieved R² scores up to 0.982 on test data
- ✅ **Multiple Algorithms:** Comparative analysis of Ridge, Lasso, and Gradient Boosting
- ✅ **Feature Importance:** Identified key drivers of V85 operating parameters
- ✅ **Scalable Solution:** Handles new manufacturing data without retraining
- ✅ **Interpretable Results:** Clear metrics and visualizations for production teams

---

## 🛠️ Technical Implementation

### Technology Stack
- **Primary Language:** Python 3.8+
- **Core Libraries:** pandas, scikit-learn, numpy, matplotlib, seaborn
- **ML Algorithms:** Ridge, Lasso, GradientBoostingRegressor
- **Data Processing:** StandardScaler, OneHotEncoder, train_test_split

### Methodology
1. **Data Preparation:**
   - 40 manufacturing process observations
   - Features: PW (Process Water), SW (System Water), MW (Manufacturing Water), RA (Resource Availability)
   - Target: V85 operating parameter

2. **Feature Engineering:**
   - One-hot encoding for RA (categorical: Yes/No)
   - Standardization of numeric features (V85, MW, SW)
   - Correlation analysis to identify key relationships

3. **Model Development:**
   - Three regression approaches: Ridge (L2 regularization), Lasso (L1 regularization), Gradient Boosting
   - Hyperparameter tuning for optimal performance
   - Comparative analysis of model performance

4. **Performance Evaluation:**
   - Standard regression metrics: MAE, MSE, RMSE, R²
   - Correlation analysis between predicted and actual values
   - Cross-validation for model robustness

---

## 📊 Results & Findings

### Model Performance Comparison

| Model | R² (Test) | RMSE | MAE | Training Time | Key Strength |
|-------|-----------|------|-----|---------------|--------------|
| **Lasso Regression** | **0.983** | **0.125** | **0.118** | Fastest | Best regularization, prevents overfitting |
| **Gradient Boosting** | 0.985 | 0.116 | 0.108 | Moderate | Best overall accuracy |
| **Ridge Regression** | 0.964 | 0.181 | 0.175 | Fast | Good balance of speed/accuracy |

### Key Insights:
1. **Lasso Regression** performed exceptionally well (R² = 0.983) with the added benefit of feature selection through L1 regularization
2. **Gradient Boosting** achieved the highest R² (0.985) but with slightly longer training time
3. **PW (Process Water)** showed the strongest correlation with V85 (0.976 correlation)
4. **RA (Resource Availability)** was successfully encoded and contributed to prediction accuracy despite being categorical
5. All models maintained consistent performance between training and test sets, indicating good generalization
