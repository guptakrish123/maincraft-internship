# 🏠 House Price Prediction using Machine Learning

This repository contains my internship projects completed as part of the  
**Artificial Intelligence & Machine Learning Internship at Maincrafts Technology**.

The project focuses on building, improving, and comparing Machine Learning models
to predict house prices using the **California Housing Dataset**.

---

## 📌 Internship Tasks Overview

### 🔹 Task 1: Linear Regression – House Price Prediction
- Built a baseline Linear Regression model
- Learned the complete ML workflow from data loading to evaluation

### 🔹 Task 2: Feature Engineering & Model Comparison
- Applied feature scaling
- Trained multiple regression models
- Compared performance and selected the best model

---

## 📊 Dataset

- **California Housing Dataset**
- Source: Built-in dataset from `scikit-learn`

**Target Variable:**  
- `MedHouseVal` / `HousePrice` (Median House Value)

**Input Features Include:**  
- Median Income  
- House Age  
- Average Rooms  
- Population & Location-based features  

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook / VS Code

---

## ⚙️ Task 1: Linear Regression (Baseline Model)

### 🔸 Objective
- Understand the complete Machine Learning workflow
- Build and evaluate a Linear Regression model

### 🔸 Steps Performed
1. Imported required libraries
2. Loaded California Housing dataset
3. Performed Exploratory Data Analysis (EDA)
4. Split data into training and testing sets
5. Trained Linear Regression model
6. Evaluated model using:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score
7. Visualized Actual vs Predicted house prices

### 🔸 Result
- Linear Regression provided a reasonable baseline for house price prediction.
- Evaluation metrics and plots are included in the notebook.

---

## ⚙️ Task 2: Feature Engineering & Model Comparison

### 🔸 Objective
- Improve model performance using preprocessing techniques
- Compare multiple Machine Learning models

### 🔸 Steps Performed
1. Feature scaling using `StandardScaler`
2. Train–test split on scaled data
3. Trained multiple models:
   - Linear Regression
   - Ridge Regression
   - Decision Tree Regressor
4. Evaluated models using RMSE and R² score
5. Created a performance comparison table
6. Selected the best-performing model
7. Visualized Actual vs Predicted values
8. Saved the best model using `joblib`

### 🔸 Result
- Model comparison helped identify the best-performing algorithm.
- Feature scaling improved training stability and performance.

---

## 📈 Model Evaluation Metrics

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **R² Score** – Coefficient of Determination  

Lower RMSE and higher R² indicate better model performance.

---

## 📂 Files in Repository

- `task1_ml_linear_regression.ipynb`  
  → Task-1: Linear Regression baseline model  

- `AI_ML_Task2_Model_Comparison.ipynb`  
  → Task-2: Feature scaling, model comparison, and optimization  

- `house_price_model.pkl` / `best_house_price_model.joblib`  
  → Saved trained model (optional)

- `README.md`  
  → Project documentation

---

## 🚀 Future Improvements

- Apply advanced models like Random Forest or XGBoost
- Hyperparameter tuning
- Cross-validation
- Deploy the model using Flask or Streamlit
- Build a simple web-based prediction interface

---

## 👨‍💻 Author

**Krish Gupta**  
Intern – Artificial Intelligence & Machine Learning  
Maincrafts Technology
