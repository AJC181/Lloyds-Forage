import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor (XGBoost)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the features that the model expects, in the exact order.
# This order is crucial for the model's performance.
MODEL_FEATURES = [
    'LoginFrequency', 'Age', 'AmountSpent', 'IncomeLevel_Medium',
    'ServiceUsage_Online Banking', 'IncomeLevel_Low', 'MaritalStatus_Widowed',
    'MaritalStatus_Single', 'MaritalStatus_Married', 'ServiceUsage_Website',
    'InteractionType_Feedback', 'ResolutionStatus_Unresolved',
    'ResolutionStatus_Resolved', 'InteractionType_Complaint',
    'InteractionType_Inquiry', 'Gender_F', 'Gender_M'
]

# --- Model Loading and Training (Cached) ---

@st.cache_data
def load_data():
    """Loads the pre-processed data."""
    try:
        # NOTE: Assumes 'Customer_Churn_Processed.csv' is in the same directory.
        data = pd.read_csv('Customer_Churn_Processed.csv')
        # Drop the CustomerID column if present, and remove ChurnStatus for X
        X = data.drop(columns=['CustomerID', 'ChurnStatus'], errors='ignore')
        y = data['ChurnStatus']
        return X, y
    except FileNotFoundError:
        st.error("Error: 'Customer_Churn_Processed.csv' not found. Please ensure the file is in the same directory as the script.")
        return None, None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None

@st.cache_resource
def train_and_tune_model(X, y):
    """
    Trains and tunes the XGBoost model using the approach defined in the project notebook.
    This function will run only once due to st.cache_resource.
    
    NOTE: Changed n_jobs from -1 to 1 to fix the 'un-serialize' error 
    related to parallel processing within Streamlit's caching mechanism.
    """
    st.info("Training and tuning the XGBoost Model... This happens only once.")

    # 1. Split data (using the same random state for reproducibility as a best practice)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Calculate scale_pos_weight for imbalance handling (4:1 ratio of No Churn to Churn)
    count_neg = y_train.value_counts()[0]
    count_pos = y_train.value_counts()[1]
    scale_pos_weight = count_neg / count_pos
    
    # 3. Define the parameter space for Randomized Search
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 0.5)
    }

    # 4. Initialize XGBClassifier with imbalance handling
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # 5. Perform Randomized Search CV
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=50,  # Number of parameter settings that are sampled
        scoring='roc_auc',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=1  # FIX: Set n_jobs to 1 to prevent the pickling error
    )
    random_search.fit(X_train, y_train)

    st.success("Model Training and Tuning Complete!")
    
    # Return the best estimator and the training data for scaling constants
    return random_search.best_estimator_, X_train

# --- Prediction and Preprocessing Logic ---

def preprocess_input(input_data, X_train):
    """
    Applies the necessary One-Hot Encoding and Standardization to the raw user input.
    """
    # 1. Create a DataFrame from the raw input
    df_raw = pd.DataFrame([input_data])

    # 2. Define all possible categories (needed for consistent OHE)
    cat_features = {
        'Gender': ['M', 'F'],
        'MaritalStatus': ['Single', 'Married', 'Divorced', 'Widowed'],
        'IncomeLevel': ['Low', 'Medium', 'High'],
        'ServiceUsage': ['Mobile App', 'Online Banking', 'Website'],
        'InteractionType': ['Inquiry', 'Complaint', 'Feedback'],
        'ResolutionStatus': ['Resolved', 'Unresolved']
    }

    # 3. Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df_raw, columns=cat_features.keys(), drop_first=False)

    # 4. Filter for the exact OHE features the model expects and ensure all are present
    final_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
    
    # Add standardized numerical features back later
    numerical_cols = ['LoginFrequency', 'Age', 'AmountSpent']
    
    # Map the encoded columns to the final structure
    for col in df_encoded.columns:
        if col in MODEL_FEATURES:
            final_input[col] = df_encoded[col].iloc[0]

    # Handle missing OHE columns by setting them to 0 (which they already are in final_input)
    # The columns that were dropped in the original processing (e.g., IncomeLevel_High) are correctly omitted.
    
    # 5. Standardize Numerical Features
    # Recalculate mean and std from the training data for a fresh StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    
    # Apply transformation
    scaled_values = scaler.transform(df_raw[numerical_cols])
    
    # Update the final input DataFrame with scaled values
    final_input[numerical_cols] = scaled_values
    
    # 6. Reorder columns to match the model's training order
    final_input = final_input[MODEL_FEATURES]

    return final_input

# --- Main App Execution ---

# Load data and train model
X, y = load_data()

if X is not None:
    try:
        model, X_train = train_and_tune_model(X, y)
    except Exception as e:
        # This error handling now catches exceptions that might occur *after* the initial import
        st.error(f"Failed to train model: {e}")
        model = None
        X_train = None
else:
    model = None
    X_train = None
    
# --- Streamlit UI: Title and Header ---

st.title("🛡️ XGBoost Customer Churn Predictor")
st.markdown("Enter the customer's profile details in the sidebar to predict their likelihood of churn using the optimized XGBoost model.")

# --- Streamlit UI: Sidebar for Input Widgets ---

with st.sidebar:
    st.header("👤 Customer Profile Input")
    st.markdown("---")

    # NUMERICAL INPUTS
    st.subheader("Transaction & Activity Metrics")
    login_freq = st.slider("Login Frequency (per month)", 0, 60, 20)
    amount_spent = st.number_input("Average Monthly Amount Spent ($)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)

    st.subheader("Demographics & Service")
    age = st.slider("Age", 18, 90, 45)
    gender = st.selectbox("Gender", ('M', 'F'))
    marital_status = st.selectbox("Marital Status", ('Single', 'Married', 'Divorced', 'Widowed'))
    income_level = st.selectbox("Income Level", ('Low', 'Medium', 'High'))
    service_usage = st.selectbox("Primary Service Usage", ('Mobile App', 'Online Banking', 'Website'))

    st.subheader("Service Interaction")
    interaction_type = st.selectbox("Most Recent Interaction Type", ('Inquiry', 'Complaint', 'Feedback'))
    resolution_status = st.selectbox("Last Interaction Resolution", ('Resolved', 'Unresolved'))

    st.markdown("---")
    predict_button = st.button("Predict Churn Risk", type="primary")

# --- Streamlit UI: Results Section ---

if predict_button and model is not None:
    # 1. Gather raw inputs
    raw_input = {
        'LoginFrequency': login_freq,
        'Age': age,
        'AmountSpent': amount_spent,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'IncomeLevel': income_level,
        'ServiceUsage': service_usage,
        'InteractionType': interaction_type,
        'ResolutionStatus': resolution_status,
    }

    # 2. Preprocess
    with st.spinner('Preprocessing data and running prediction...'):
        processed_df = preprocess_input(raw_input, X_train)

    # 3. Predict Probability
    prediction_proba = model.predict_proba(processed_df)[:, 1][0]
    prediction_status = 'CHURN RISK' if prediction_proba >= 0.5 else 'LOW RISK'
    
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Predicted Churn Status", value=prediction_status)
        st.markdown(f"The model predicts a **{prediction_status}** for this customer profile.")

    with col2:
        # Visualize the probability with a circular gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_proba * 100,
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 25], 'color': "green"},
                    {'range': [25, 50], 'color': "yellowgreen"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}
        ))
        
        fig.update_layout(height=250, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Risk Interpretation")

    if prediction_status == 'CHURN RISK':
        st.error(f"**High Risk ({prediction_proba:.2%})**")
        st.markdown(
            """
            This profile shows characteristics highly correlated with customer churn. 
            **Recommended Action:** Immediate intervention is required. Consider targeted retention offers, 
            a personalized follow-up from a senior service agent, or a high-value incentive.
            """
        )
    else:
        st.success(f"**Low Risk ({prediction_proba:.2%})**")
        st.markdown(
            """
            This customer profile is currently considered low risk for churn. 
            **Recommended Action:** Continue standard engagement protocols. 
            Monitor interaction frequency and transaction activity for any sudden drop-offs.
            """
        )

# Add a warning if model training failed
elif model is None:
    st.warning("Please address the data loading/training error displayed above to use the prediction feature.")
