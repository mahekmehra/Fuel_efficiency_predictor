# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, confusion_matrix

# Load the trained model and test data
model = joblib.load('xgboost_mpg_model.pkl')
df = pd.read_csv('vehicle_data.csv')  # Original dataset for context

# Define feature lists (must match training)
numeric_features = ['engine_size', 'cylinders', 'horsepower', 'torque', 'weight', 
                    'drag_coefficient', 'acceleration_time', 'power_to_weight', 
                    'torque_to_weight', 'engine_efficiency']
categorical_features = ['transmission', 'drivetrain', 'tire_type', 'fuel_type', 
                        'fuel_injection', 'turbocharged', 'hybrid_system']
target = 'mpg'

# Recreate engineered features for the loaded dataset
df['power_to_weight'] = df['horsepower'] / df['weight']
df['torque_to_weight'] = df['torque'] / df['weight']
df['engine_efficiency'] = df['horsepower'] / df['engine_size'].replace(0, 0.1)

# Precompute test predictions and R² score
X_test = df[numeric_features + categorical_features]
y_test = df[target]
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

# Streamlit UI
st.set_page_config(page_title="Vehicle Fuel Efficiency Predictor", page_icon="🚗", layout="wide")

# Title and Introduction
st.title("🚗 Vehicle Fuel Efficiency Predictor")
st.markdown("""
    Welcome to the Vehicle MPG Predictor! This tool uses an **XGBoost** machine learning model to predict your vehicle's fuel efficiency (MPG) based on its specifications.
""")

# Tabs for different sections
tab1, tab2 = st.tabs(["Prediction", "Model Details"])

# Tab 1: Prediction Interface
with tab1:
    st.header("Predict Your Vehicle's MPG")
    st.markdown("Enter your vehicle's specifications and get an instant MPG prediction.")

    # Sidebar for user inputs
    st.sidebar.header("Vehicle Specifications")

    # Numeric inputs
    engine_size = st.sidebar.slider("Engine Size (L)", 0.0, 5.0, 2.0, 0.1)
    cylinders = st.sidebar.slider("Cylinders", 0, 12, 4, 1)
    horsepower = st.sidebar.slider("Horsepower", 50.0, 500.0, 200.0, 5.0)
    torque = st.sidebar.slider("Torque (lb-ft)", 50.0, 600.0, 250.0, 5.0)
    weight = st.sidebar.slider("Weight (lbs)", 1000.0, 4000.0, 3000.0, 50.0)
    drag_coefficient = st.sidebar.slider("Drag Coefficient", 0.20, 0.40, 0.30, 0.01)
    acceleration_time = st.sidebar.slider("0-60 mph Time (s)", 5.0, 20.0, 10.0, 0.5)

    # Categorical inputs
    transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "CVT", "DCT"])
    drivetrain = st.sidebar.selectbox("Drivetrain", ["FWD", "RWD", "AWD"])
    tire_type = st.sidebar.selectbox("Tire Type", ["AllTerrain", "Performance", "LowRolling"])
    fuel_type = st.sidebar.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])
    fuel_injection = st.sidebar.selectbox("Fuel Injection", ["Direct", "Port", "Carburetor"])
    turbocharged = st.sidebar.checkbox("Turbocharged")
    hybrid_system = st.sidebar.checkbox("Hybrid System")

    # Prepare input data
    input_data = {
        'engine_size': engine_size,
        'cylinders': cylinders,
        'horsepower': horsepower,
        'torque': torque,
        'weight': weight,
        'drag_coefficient': drag_coefficient,
        'acceleration_time': acceleration_time,
        'transmission': transmission,
        'drivetrain': drivetrain,
        'tire_type': tire_type,
        'fuel_type': fuel_type,
        'fuel_injection': fuel_injection,
        'turbocharged': 1 if turbocharged else 0,
        'hybrid_system': 1 if hybrid_system else 0
    }

    # Feature engineering for user input
    input_data['power_to_weight'] = input_data['horsepower'] / input_data['weight']
    input_data['torque_to_weight'] = input_data['torque'] / input_data['weight']
    input_data['engine_efficiency'] = input_data['horsepower'] / (input_data['engine_size'] if input_data['engine_size'] > 0 else 0.1)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction
    if st.sidebar.button("Predict MPG"):
        prediction = model.predict(input_df)[0]
        
        # Main content with aligned graphs
        st.subheader("Prediction Results")
        
        # Display prediction and custom message side by side
        col_pred, col_msg = st.columns([1, 2])
        with col_pred:
            st.metric("Predicted MPG", f"{prediction:.2f}")
        with col_msg:
            if prediction < 40:
                st.warning("Low Fuel Efficiency: Below 40 MPG, this vehicle might lead to higher fuel costs.")
            elif 40 <= prediction <= 60:
                st.success("Average Fuel Efficiency: Between 40-60 MPG, this car offers good value and helps keep fuel costs low.")
            else:
                st.success("Impressive Fuel Efficiency: Above 60 MPG, this vehicle is exceptionally efficient—great job!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # MPG gauge visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            gauge = np.linspace(0, 120, 100)
            ax.plot(gauge, np.sin(np.linspace(0, np.pi, 100)), 'g-', lw=5)
            ax.fill_between(gauge, 0, np.sin(np.linspace(0, np.pi, 100)), alpha=0.2)
            ax.axvline(x=prediction, color='red', linestyle='--', label=f'Predicted: {prediction:.2f}')
            ax.set_ylim(0, 1.5)
            ax.set_xlim(0, 120)
            ax.set_title("MPG Gauge")
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            # Prediction Distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['mpg'], bins=30, kde=True, ax=ax, color='blue', alpha=0.5)
            ax.axvline(prediction, color='red', linestyle='--', label=f'Your Prediction: {prediction:.2f}')
            ax.set_title("MPG Distribution")
            ax.set_xlabel("MPG")
            ax.legend()
            st.pyplot(fig)

# Tab 2: Model Details
with tab2:
    st.header("About Our Model")
    st.markdown("""
        ### Model Training Details
        Our MPG prediction model is built using **XGBoost**, a powerful gradient boosting algorithm known for its accuracy and efficiency. Here's how it was trained:
        
        - **Dataset**: Trained on a comprehensive vehicle dataset with features like engine size, horsepower, weight, and more.
        - **Feature Engineering**: Added derived features such as power-to-weight ratio, torque-to-weight ratio, and engine efficiency.
        - **Preprocessing**: Numerical features were scaled using StandardScaler; categorical features were one-hot encoded.
        - **Hyperparameter Tuning**: Used GridSearchCV with 5-fold cross-validation to optimize parameters like `n_estimators`, `max_depth`, `learning_rate`, etc.
        - **Objective**: Minimized squared error for regression (`reg:squarederror`).

        ### Model Performance
        The model achieved the following performance on the entire dataset:
    """)
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("Accuracy", f"{r2*100:.2f}%")
    st.markdown(f"This means the model explains {r2*100:.2f}% of the variance in MPG values, indicating strong predictive power.")

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted MPG")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.5, color='blue', label='Predictions')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title('Actual vs Predicted MPG')
    ax.legend()
    st.pyplot(fig)

    # Additional Visualizations of Car Features
    st.subheader("Vehicle Feature Insights")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### MPG vs Horsepower")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='horsepower', y='mpg', hue='fuel_type', size='weight', alpha=0.6, ax=ax)
        ax.set_title("MPG vs Horsepower by Fuel Type")
        st.pyplot(fig)
    
    with col4:
        st.markdown("#### MPG by Transmission")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='transmission', y='mpg', palette='coolwarm', ax=ax)
        ax.set_title("MPG Distribution by Transmission")
        st.pyplot(fig)

    # Correlation Matrix Heatmap
    st.subheader("Correlation Matrix")
    st.markdown("This heatmap shows the correlations between numerical features in the dataset. Values close to 1 or -1 indicate strong positive or negative correlations, respectively.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    st.pyplot(fig)

    # Pseudo-Confusion Matrix for Regression
    st.subheader("Confusion Matrix Approximation")
    st.markdown("""
        **Note**: Since MPG prediction is a regression task, a traditional confusion matrix isn’t directly applicable. To provide insight, we’ve discretized the continuous MPG values into three bins (Low: <20, Medium: 20-30, High: >30) and compared actual vs predicted categories. This is an approximation to show how well predictions align with actual values in broad categories.
    """)
    
    # Discretize MPG into bins
    bins = [0, 20, 30, float('inf')]
    labels = ['Low (<20)', 'Medium (20-30)', 'High (>30)']
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
    y_pred_binned = pd.cut(y_pred_test, bins=bins, labels=labels, include_lowest=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted MPG Category')
    ax.set_ylabel('Actual MPG Category')
    ax.set_title('Confusion Matrix Approximation (Binned MPG)')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and XGBoost | © 2025 Vehicle Fuel Efficiency Predictor")