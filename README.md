# Delivery Duration Prediction (Capstone Project)

## 📌 Project Overview
This project focuses on predicting **food delivery duration** using historical, DoorDash-like data. The objective is to estimate the total time (in seconds) between order placement and actual delivery using classical regression techniques.

The project emphasizes an **end-to-end machine learning workflow**, including data understanding, preprocessing, feature engineering, model experimentation, and statistical evaluation, rather than achieving unrealistically low error metrics.

---

## 🎯 Problem Statement
Food delivery platforms require accurate delivery time estimates to maintain customer satisfaction. Incorrect predictions, especially large delays, can lead to poor user experience and customer churn.

**Goal:**  
Predict the total delivery duration (in seconds) using order, store, and marketplace-level features.

**Target Variable:**  
delivery_duration = actual_delivery_time − created_at

---

## 🧠 Project Approach
The project was approached as a real-world ML problem with the following mindset:
- No single model guarantees optimal performance
- Data limitations constrain achievable accuracy
- Understanding model behavior is more important than chasing metrics

---

## 📊 Dataset Description
Each row in the dataset represents a single food delivery order.

### Feature Categories
- **Time Features:** order timestamp, hour of day, weekend indicator
- **Order Features:** total items, subtotal, item prices
- **Store Features:** store category, order protocol
- **Marketplace Features:** active dashers, busy dashers, outstanding orders
- **System Estimates:** estimated order prep and driving durations

After preprocessing and encoding, the final dataset contains:
- **195,364 rows**
- **99 features**

---

## 🛠️ Data Preprocessing & Feature Engineering
Key preprocessing steps include:
- Timestamp parsing and delivery duration computation
- Missing value imputation using grouped medians
- Outlier awareness and log transformation for skewed features
- One-hot encoding of categorical variables
- Removal of non-informative identifiers
- Validation for missing, duplicate, and infinite values

All preprocessing steps were validated using statistical diagnostics to ensure correctness and absence of data leakage.

---

## 📈 Exploratory Data Analysis (EDA)
EDA revealed:
- High natural variability in delivery duration  
  - Mean ≈ 2859 seconds  
  - Standard deviation ≈ 1115 seconds
- Weak to moderate correlations between individual features and the target
- Skewed distributions for price- and count-based features

These observations guided realistic expectations for model performance.

---

## 🤖 Models Implemented
The following regression models were trained and evaluated:

- **Linear Regression (Baseline)**
- **Decision Tree Regressor**
- **K-Nearest Neighbors (KNN)**
- **Polynomial Regression**
- **Support Vector Regression (SVR)** *(evaluated on sampled data due to computational cost)*

All models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📊 Model Performance Summary
| Model | MAE (seconds) | R² |
|------|--------------|----|
| Linear Regression | ~700 | ~0.23 |
| Decision Tree | ~733 | ~0.17 |
| KNN | ~750–800 | ~0.00 |
| Polynomial Regression | ~754 | Negative |
| SVR (sampled) | ~753 | ~0.07 |

**Linear Regression achieved the best generalization performance** among the evaluated models.

---

## 📌 Key Insights
- Model performance is primarily constrained by **data quality and feature availability**, not preprocessing.
- Increasing model complexity did not improve results and often reduced generalization.
- The achieved MAE (~700 seconds) is significantly lower than the natural variability of delivery times, indicating meaningful signal capture.
- Simple models with strong preprocessing can outperform more complex alternatives in noisy, real-world datasets.

---

## ⚠️ Limitations
- No real-time traffic, routing, or weather data
- No driver-level or spatial features
- Only classical regression models were explored (as per course scope)

---

## ✅ Final Conclusion
This project demonstrates a complete and realistic machine learning workflow for delivery time prediction. The results highlight the importance of data understanding, feature engineering, and honest evaluation over aggressive optimization. Linear Regression, despite its simplicity, proved to be the most stable and interpretable model under the given constraints.

---
