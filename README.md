# 🚗 Fuel Efficiency Prediction using Machine Learning  
Predicting Vehicle MPG based on Driving & Vehicle Specifications  

## 📌 Overview
This project builds a **machine learning-based tool** that predicts a vehicle’s fuel efficiency (MPG) using specifications such as engine size, horsepower, weight, transmission type, fuel type, and more. The model is trained on a **hybrid dataset of 200,000 real + synthetic vehicle entries**, ensuring diversity and realism.

An **XGBoost Regression Model** is used for prediction, combined with extensive feature engineering and data preprocessing. The final model is deployed through a **Streamlit web application** for real-time user interaction.

---

## 🚀 Features
- ✔️ Predict MPG for Gasoline, Diesel, Hybrid, and Electric vehicles  
- ✔️ Synthetic + real dataset with **15 engineered features**  
- ✔️ Advanced model training with **XGBoost + GridSearchCV tuning**  
- ✔️ Interactive Streamlit interface for live predictions  
- ✔️ Feature importance & correlation insights  
- ✔️ Model evaluation using R², MAE, MSE  
- ✔️ Saved model using `joblib` for fast deployment  

---

## 📊 Dataset Details
The dataset includes **200,000 rows and 15 features**:

### **🔢 Numeric Features (10)**
- engine_size  
- cylinders  
- horsepower  
- torque  
- weight  
- acceleration_time  
- drag_coefficient  
- power_to_weight  
- torque_to_weight  
- engine_efficiency  

### **🧩 Categorical Features (5+)**
- fuel_type  
- transmission  
- drivetrain  
- tire_type  
- turbocharged / hybrid system indicators  

### **🎯 Target Variable**
- **MPG (fuel efficiency)**  

Synthetic logic ensures realistic MPG ranges:  
- Gasoline: 15–40  
- Diesel: 25–50  
- Hybrid: 40–70  
- Electric: 80–120  

---

## ⚙️ Methodology

### **1. Data Preprocessing**
- Missing value handling  
- Standard scaling for numeric features  
- One-hot encoding for categorical features  
- Outlier control and noise addition  
- Feature engineering for improved performance  

### **2. Model Training**
- XGBoost Regressor (`reg:squarederror`)  
- Hyperparameter tuning using GridSearchCV  
- 80/20 Train-Test Split  

### **3. Evaluation Metrics**
- **R² ≈ 0.87**  
- **MAE ≈ 7 MPG**  
- **MSE ≈ 68.6**  

XGBoost outperformed Linear Regression and Random Forest.

---

## 🛠️ Technologies Used
- **Python**  
- **Pandas, NumPy**  
- **Scikit-learn**  
- **XGBoost**  
- **Streamlit**  
- **Matplotlib & Seaborn**  
- **Joblib**  

---

## 🧪 Results & Insights
- Strong negative correlation: **weight vs MPG (-0.8)**  
- Positive correlation: **engine_efficiency vs MPG (+0.6)**  
- Most important features:  
  - fuel_type  
  - weight  
  - engine_size  
  - power_to_weight  

Scatterplots and feature importance graphs confirm strong predictive capability.

---

## 🏁 Conclusion
This project demonstrates a practical, high-performing ML system capable of predicting fuel efficiency with strong accuracy. The combination of synthetic data, engineered features, and XGBoost results in a scalable and deployable real-world solution.

Future improvements include:  
- Integration with real EPA datasets  
- Incorporating driving behavior logs  
- Weather, terrain, and road-type modeling  
- Deep learning extensions  

---

