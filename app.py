import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from eda.data_preprocessor import DataPreprocessor
from eda.graph_builder import GraphBuilder
from eda.evaluation import ModelEvaluator
from eda.visualization import Visualizer
from models.hybrid_model import HybridFraudDetector
from trainer import ModelTrainer

# Page configuration
st.set_page_config(
    page_title="HNN Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🔒 Hybrid Neural Network (GNN+LSTM) Credit Card Fraud Detection")
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'graph_builder' not in st.session_state:
    st.session_state.graph_builder = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

# Sidebar
import streamlit as st

# Initialize session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "Data Overview"

# Sidebar navigation with buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Data Overview"):
    st.session_state.page = "Data Overview"
if st.sidebar.button("Model Training"):
    st.session_state.page = "Model Training"
if st.sidebar.button("Model Evaluation"):
    st.session_state.page = "Model Evaluation"
if st.sidebar.button("Real-time Prediction"):
    st.session_state.page = "Real-time Prediction"
if st.sidebar.button("Performance Dashboard"):
    st.session_state.page = "Performance Dashboard"

# Set the active page
page = st.session_state.page

def calculate_rule_based_risk_score(transaction_data, user_avg_amount=200.0):
    """
    Calculate rule-based risk score using heuristics
    
    Args:
        transaction_data: Dictionary containing transaction details
        user_avg_amount: User's average transaction amount over 7 days
    
    Returns:
        risk_score: Float between 0.0 and 1.0
        risk_factors: List of triggered risk factors
    """
    risk_score = 0.0
    risk_factors = []
    
    # 🧠 User & Transaction Behavior Rules
    if transaction_data['Transaction_Amount'] > user_avg_amount * 2:
        risk_score += 0.2
        risk_factors.append("High transaction amount (>2x average)")
    
    if transaction_data['Failed_Transaction_Count_7d'] > 3:
        risk_score += 0.2
        risk_factors.append("High failed transactions (>3 in 7 days)")
    
    if transaction_data['Daily_Transaction_Count'] > 10:
        risk_score += 0.15
        risk_factors.append("High daily transaction count (>10)")
    
    if transaction_data['Transaction_Distance'] > 500:
        risk_score += 0.2
        risk_factors.append("High transaction distance (>500km)")
    
    # Convert card age from months to check if < 6 months
    card_age_months = transaction_data['Card_Age'] / 30  # Assuming card_age is in days
    if card_age_months < 6:
        risk_score += 0.15
        risk_factors.append("New card (<6 months old)")
    
    if transaction_data['Account_Balance'] < transaction_data['Transaction_Amount']:
        risk_score += 0.2
        risk_factors.append("Insufficient account balance")
    
    # 📍 Location & Device Rules
    unusual_locations = ['Tokyo', 'Sydney', 'Mumbai']  # Example unusual locations
    if transaction_data['Location'] in unusual_locations:
        risk_score += 0.2
        risk_factors.append("Unusual location")
    
    if transaction_data['Device_Type'] in ['Tablet']:  # Treating tablet as new/unrecognized
        risk_score += 0.2
        risk_factors.append("New/Unrecognized device type")
    
    if transaction_data['Authentication_Method'] in ['OTP', 'Password']:
        risk_score += 0.15
        risk_factors.append("Weak authentication method")
    
    if transaction_data['Transaction_Type'] == 'Online':
        risk_score += 0.1
        risk_factors.append("Online transaction")
    
    # ⚠️ High-Risk Flags
    if transaction_data['IP_Address_Flag']:
        risk_score += 0.3
        risk_factors.append("Suspicious IP address")
    
    if transaction_data['Previous_Fraudulent_Activity']:
        risk_score += 0.4
        risk_factors.append("Previous fraudulent activity")
    
    if transaction_data['Is_Weekend']:
        risk_score += 0.1
        risk_factors.append("Weekend transaction")
    
    # Cap the risk score at 1.0
    pre_risk_score = min(risk_score, 1.0)
    
    return pre_risk_score, risk_factors


def load_data():
    """Load and reduce the dataset to a maximum of 10,000 rows"""
    if 'data' not in st.session_state:
        try:
            # Load the full dataset
            data = pd.read_csv('dataset/synthetic_fraud_dataset.csv')

            # Limit to 10,000 rows if larger
            if len(data) > 10000:
                data = data.sample(n=5000, random_state=42).reset_index(drop=True)

            st.session_state.data = data
            st.success(f"Dataset loaded successfully! Shape: {data.shape}")
        except FileNotFoundError:
            st.error("Dataset file not found. Please ensure the CSV file is in the correct location.")
            st.stop()
    return st.session_state.data

# Data Overview Page
if page == "Data Overview":
    st.header("📊 Dataset Overview")
    
    data = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(data):,}")
    with col2:
        fraud_count = data['Fraud_Label'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    with col3:
        fraud_rate = (fraud_count / len(data)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with col4:
        st.metric("Features", len(data.columns) - 1)
    
    # Dataset preview
    st.subheader("Dataset Sample")
    st.dataframe(data.head(10))
    
    # Data distribution visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud Distribution")
        fraud_counts = data['Fraud_Label'].value_counts()
        fig = px.pie(values=fraud_counts.values, names=['Legitimate', 'Fraudulent'], 
                     title="Transaction Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(data, x='Transaction_Amount', color='Fraud_Label',
                          title="Transaction Amount by Fraud Status", nbins=50)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlations
    st.subheader("Feature Correlations with Fraud")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlations = data[numeric_cols].corr()['Fraud_Label'].sort_values(ascending=False)
    
    fig = px.bar(x=correlations.values[1:-1], y=correlations.index[1:-1],
                 title="Feature Correlations with Fraud Label", orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "Model Training":
    st.header("🧠 Model Training")
    
    data = load_data()
    
    # Training parameters
    st.subheader("Training Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Number of Epochs", 5, 100, 10)
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    
    with col2:
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
        hidden_dim = st.slider("Hidden Dimension", 32, 256, 128)
    
    with col3:
        gnn_layers = st.slider("GNN Layers", 1, 5, 2)
        lstm_layers = st.slider("LSTM Layers", 1, 3, 2)
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    col_a, col_b = st.columns(2)
    with col_a:
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        use_smote = st.checkbox("Use SMOTE for Class Balancing", value=True)
    
    with col_b:
        test_size = st.slider("Test Split Size", 0.1, 0.3, 0.2)
        random_state = st.number_input("Random State", value=42)
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training model... This may take several minutes."):
            try:
                # Initialize preprocessor
                preprocessor = DataPreprocessor()
                
                # Preprocess data
                st.info("Preprocessing data...")
                X_processed, y = preprocessor.preprocess(data)
                
                # Build graph
                st.info("Building transaction graph...")
                graph_builder = GraphBuilder()
                graph_data = graph_builder.build_graph(data, X_processed)
                
                # Initialize trainer
                trainer = ModelTrainer(
                    hidden_dim=hidden_dim,
                    gnn_layers=gnn_layers,
                    lstm_layers=lstm_layers,
                    dropout=dropout,
                    learning_rate=learning_rate
                )
                
                # Train model
                st.info("Training hybrid model...")
                model, history = trainer.train(
                    graph_data, X_processed, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    test_size=test_size,
                    use_smote=use_smote,
                    random_state=random_state
                )
                
                # Store in session state
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor
                st.session_state.graph_builder = graph_builder
                st.session_state.training_history = history
                st.session_state.model_trained = True
                
                st.success("Model training completed successfully!")
                
                # Display training results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Training Loss")
                    if 'train_loss' in history and len(history['train_loss']) > 0:
                        fig = px.line(x=list(range(len(history['train_loss']))), y=history['train_loss'],
                                     title="Training Loss Over Time")
                        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No training loss data available")
                
                with col2:
                    st.subheader("Validation Metrics")
                    if all(key in history for key in ['val_f1', 'val_precision', 'val_recall']) and len(history['val_f1']) > 0:
                        metrics_df = pd.DataFrame({
                            'Epoch': list(range(len(history['val_f1']))),
                            'F1 Score': history['val_f1'],
                            'Precision': history['val_precision'],
                            'Recall': history['val_recall']
                        })
                        
                        fig = px.line(metrics_df, x='Epoch', y=['F1 Score', 'Precision', 'Recall'],
                                     title="Validation Metrics Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No validation metrics data available")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")

    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Model Training' page.")
    else:
        data = load_data()
        model = st.session_state.model
        preprocessor = st.session_state.preprocessor
        graph_builder = st.session_state.graph_builder

        # Evaluate model
        evaluator = ModelEvaluator()

        with st.spinner("Evaluating model performance..."):
            try:
                # Preprocess data
                X_processed, y = preprocessor.preprocess(data)
                graph_data = graph_builder.build_graph(data, X_processed)

                # Prepare sequence data for evaluation
                def prepare_sequence_data_and_labels(X, y, sequence_length=10):
                    """Prepare sequence data for LSTM and align labels"""
                    sequences = []
                    aligned_labels = []
                    
                    for i in range(len(X) - sequence_length + 1):
                        sequences.append(X[i:i+sequence_length])
                        aligned_labels.append(y[i + sequence_length - 1])  # Use the label of the last item in sequence
                    
                    # If we don't have enough data for sequences, use all data
                    if len(sequences) == 0:
                        sequences = [np.tile(X, (sequence_length, 1))[:sequence_length] for _ in range(len(X))]
                        aligned_labels = y.tolist()
                    
                    return torch.FloatTensor(np.array(sequences)), np.array(aligned_labels)
                
                # Prepare sequence data and aligned labels
                sequence_data, y_aligned = prepare_sequence_data_and_labels(X_processed, y)
                
                # Make predictions
                predictions, probabilities = evaluator.evaluate_model(
                    model, graph_data, sequence_data, y_aligned
                )

                # Calculate metrics using aligned labels
                f1 = f1_score(y_aligned, predictions)
                auc = roc_auc_score(y_aligned, probabilities)
                cm = confusion_matrix(y_aligned, predictions)
                report = classification_report(y_aligned, predictions, output_dict=True)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("F1 Score", f"{f1:.4f}")
                with col2:
                    st.metric("AUC Score", f"{auc:.4f}")
                with col3:
                    st.metric("Precision", f"{report['1']['precision']:.4f}")
                with col4:
                    st.metric("Recall", f"{report['1']['recall']:.4f}")

                # Visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)

                with col2:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = evaluator.calculate_roc_curve(y_aligned, probabilities)
                    fig = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {auc:.4f})')
                    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, 
                                 line=dict(dash='dash', color='red'))
                    fig.update_layout(xaxis_title="False Positive Rate", 
                                    yaxis_title="True Positive Rate")
                    st.plotly_chart(fig, use_container_width=True)

                # Classification report
                st.subheader("Detailed Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

# Real-time Prediction Page
elif page == "Real-time Prediction":
    st.header("🔍 Real-time Fraud Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Model Training' page.")
    else:
        st.subheader("Enter Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_id = st.text_input("User ID", value="USER_1234")
            amount = st.number_input("Transaction Amount", min_value=0.01, value=100.0)
            transaction_type = st.selectbox("Transaction Type", 
                                          ["POS", "Online", "ATM Withdrawal", "Bank Transfer"])
            device_type = st.selectbox("Device Type", ["Mobile", "Laptop", "Tablet"])
        
        with col2:
            location = st.selectbox("Location", ["New York", "London", "Tokyo", "Sydney", "Mumbai"])
            merchant_category = st.selectbox("Merchant Category", 
                                           ["Groceries", "Electronics", "Travel", "Restaurants", "Clothing"])
            card_type = st.selectbox("Card Type", ["Visa", "Mastercard", "Amex", "Discover"])
            authentication = st.selectbox("Authentication Method", 
                                        ["PIN", "OTP", "Biometric", "Password"])
        
        with col3:
            account_balance = st.number_input("Account Balance", min_value=0.0, value=5000.0)
            daily_txn_count = st.slider("Daily Transaction Count", 1, 20, 5)
            card_age = st.slider("Card Age (months)", 1, 300, 60)
            
        
        # Additional features
        with st.expander("Additional Features"):
            col1, col2 = st.columns(2)
            with col1:
                ip_flag = st.checkbox("Suspicious IP Address")
                prev_fraud = st.checkbox("Previous Fraudulent Activity")
                is_weekend = st.checkbox("Weekend Transaction")
            with col2:
                avg_amount_7d = st.number_input("Average Amount (7 days)", value=200.0)
                failed_count_7d = st.slider("Failed Transactions (7 days)", 0, 10, 2)
                transaction_distance = st.number_input("Transaction Distance (km)", value=100.0)
        
        if st.button("Predict Fraud", type="primary"):
            try:
                # Create transaction data
                transaction_data = {
                    'User_ID': user_id,
                    'Transaction_Amount': amount,
                    'Transaction_Type': transaction_type,
                    'Device_Type': device_type,
                    'Location': location,
                    'Merchant_Category': merchant_category,
                    'Card_Type': card_type,
                    'Authentication_Method': authentication,
                    'Account_Balance': account_balance,
                    'Daily_Transaction_Count': daily_txn_count,
                    'Card_Age': card_age,
                    'IP_Address_Flag': int(ip_flag),
                    'Previous_Fraudulent_Activity': int(prev_fraud),
                    'Is_Weekend': int(is_weekend),
                    'Avg_Transaction_Amount_7d': avg_amount_7d,
                    'Failed_Transaction_Count_7d': failed_count_7d,
                    'Transaction_Distance': transaction_distance,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Calculate rule-based risk score
                calculated_risk_score, risk_factors = calculate_rule_based_risk_score(
                    transaction_data, avg_amount_7d
                )
                transaction_data['Risk_Score'] = calculated_risk_score


                # Make prediction
                model = st.session_state.model
                preprocessor = st.session_state.preprocessor

                # Convert to DataFrame for preprocessing
                transaction_df = pd.DataFrame([transaction_data])

                # Preprocess
                X_processed, _ = preprocessor.preprocess(transaction_df, is_prediction=True)

                # Simple prediction (without graph for single transaction)
                model.eval()
                with torch.no_grad():
                    # Use only LSTM component for single transaction prediction
                    X_tensor = torch.FloatTensor(X_processed)
                    # Get sequence features for LSTM
                    seq_features = X_tensor.unsqueeze(0)  # Add batch dimension

                    # Forward pass through LSTM only
                    lstm_out = model.lstm_model(seq_features)  # Shape: [1, hidden_dim]
                    
                    # Apply a final linear layer to get single output
                    # Create a temporary linear layer for classification
                    if not hasattr(model, 'temp_classifier'):
                        model.temp_classifier = nn.Linear(lstm_out.size(1), 1)
                        if torch.cuda.is_available():
                            model.temp_classifier = model.temp_classifier.cuda()
                    
                    # Get final prediction
                    logits = model.temp_classifier(lstm_out)  # Shape: [1, 1]
                    prediction_prob = torch.sigmoid(logits).item()  # Now this works
                    prediction = 1 if prediction_prob > 0.5 else 0

                # Display results prominently
                st.subheader("🎯 Prediction Results")
                
                # Main results in prominent cards
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 🤖 ML Model Prediction")
                    if prediction == 1:
                        st.error("🚨 **FRAUD DETECTED**")
                        st.error(f"**Probability: {prediction_prob:.4f}**")
                    else:
                        st.success("✅ **LEGITIMATE**")
                        st.success(f"**Probability: {prediction_prob:.4f}**")

                with col2:
                    st.markdown("### 📊 Rule-Based Risk Score")
                    risk_level = "🔴 HIGH" if calculated_risk_score > 0.7 else "🟡 MEDIUM" if calculated_risk_score > 0.4 else "🟢 LOW"
                    
                    if calculated_risk_score > 0.7:
                        st.error(f"**Risk Score: {calculated_risk_score:.3f}**")
                        st.error(f"**Risk Level: {risk_level}**")
                    elif calculated_risk_score > 0.4:
                        st.warning(f"**Risk Score: {calculated_risk_score:.3f}**")
                        st.warning(f"**Risk Level: {risk_level}**")
                    else:
                        st.success(f"**Risk Score: {calculated_risk_score:.3f}**")
                        st.success(f"**Risk Level: {risk_level}**")

                with col3:
                    st.markdown("### 🎯 Combined Assessment")
                    combined_score = (prediction_prob + calculated_risk_score) / 2
                    if combined_score > 0.6:
                        final_decision = "🚨 **HIGH RISK**"
                        st.error(final_decision)
                        st.error(f"**Combined Score: {combined_score:.3f}**")
                    elif combined_score > 0.3:
                        final_decision = "⚠️ **MEDIUM RISK**"
                        st.warning(final_decision)
                        st.warning(f"**Combined Score: {combined_score:.3f}**")
                    else:
                        final_decision = "✅ **LOW RISK**"
                        st.success(final_decision)
                        st.success(f"**Combined Score: {combined_score:.3f}**")

                # Detailed gauge visualizations
                st.subheader("📈 Score Visualizations")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # ML Probability gauge
                    fig_ml = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "ML Fraud Probability"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "red" if prediction_prob > 0.5 else "green"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgreen"},
                                {'range': [0.5, 1], 'color': "lightcoral"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5}}))
                    fig_ml.update_layout(height=300)
                    st.plotly_chart(fig_ml, use_container_width=True)

                with col2:
                    # Rule-based risk gauge
                    fig_rule = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = calculated_risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Rule-Based Risk Score"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "red" if calculated_risk_score > 0.7 else "orange" if calculated_risk_score > 0.4 else "green"},
                            'steps': [
                                {'range': [0, 0.4], 'color': "lightgreen"},
                                {'range': [0.4, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "lightcoral"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7}}))
                    fig_rule.update_layout(height=300)
                    st.plotly_chart(fig_rule, use_container_width=True)

                with col3:
                    # Combined score gauge
                    fig_combined = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = combined_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Combined Risk Score"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "red" if combined_score > 0.6 else "orange" if combined_score > 0.3 else "green"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgreen"},
                                {'range': [0.3, 0.6], 'color': "yellow"},
                                {'range': [0.6, 1], 'color': "lightcoral"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.6}}))
                    fig_combined.update_layout(height=300)
                    st.plotly_chart(fig_combined, use_container_width=True)
                
                # Score breakdown summary
                st.subheader("📋 Score Breakdown")
                breakdown_col1, breakdown_col2 = st.columns(2)
                
                with breakdown_col1:
                    st.info("**Risk Composition Details:**")
                    st.write(f"🤖 **ML Model Score:** {prediction_prob:.3f}")
                    st.write(f"📊 **Rule-Based Score:** {calculated_risk_score:.3f}")
                    st.write(f"🎯 **Combined Score:** {combined_score:.3f}")
                    st.write(f"🏁 **Final Decision:** {final_decision}")

                with breakdown_col2:
                    st.info("**Decision Thresholds:**")
                    st.write("🟢 **Low Risk:** < 0.3")
                    st.write("🟡 **Medium Risk:** 0.3 - 0.6")
                    st.write("🔴 **High Risk:** > 0.6")
                    st.write(f"📊 **Rule Factors Triggered:** {len(risk_factors)}")

                # Detailed risk factors analysis
                st.subheader("Triggered Risk Factors")
                
                if risk_factors:
                    # Create tabs for different risk categories
                    risk_tabs = st.tabs(["🧠 Behavior", "📍 Location & Device", "⚠️ High-Risk Flags"])
                    
                    behavior_factors = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['amount', 'failed', 'daily', 'distance', 'card', 'balance'])]
                    location_factors = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['location', 'device', 'authentication', 'online'])]
                    high_risk_factors = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['ip', 'previous', 'weekend'])]
                    
                    with risk_tabs[0]:
                        if behavior_factors:
                            for factor in behavior_factors:
                                st.warning(f"⚠️ {factor}")
                        else:
                            st.success("✅ No behavioral risk factors detected")
                    
                    with risk_tabs[1]:
                        if location_factors:
                            for factor in location_factors:
                                st.warning(f"⚠️ {factor}")
                        else:
                            st.success("✅ No location/device risk factors detected")
                    
                    with risk_tabs[2]:
                        if high_risk_factors:
                            for factor in high_risk_factors:
                                st.error(f"🚨 {factor}")
                        else:
                            st.success("✅ No high-risk flags detected")
                    
                    # Risk mitigation suggestions
                    st.subheader("Risk Mitigation Recommendations")
                    
                    if calculated_risk_score > 0.7:
                        st.error("**HIGH RISK TRANSACTION - Immediate Action Required:**")
                        st.write("• Block transaction and contact customer")
                        st.write("• Require additional authentication")
                        st.write("• Manual review by fraud analyst")
                    elif calculated_risk_score > 0.4:
                        st.warning("**MEDIUM RISK TRANSACTION - Enhanced Monitoring:**")
                        st.write("• Step-up authentication required")
                        st.write("• Monitor for related transactions")
                        st.write("• SMS/Email verification")
                    else:
                        st.success("**LOW RISK TRANSACTION - Standard Processing:**")
                        st.write("• Process transaction normally")
                        st.write("• Continue standard monitoring")
                else:
                    st.success("✅ No risk factors detected - Transaction appears legitimate")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Performance Dashboard Page
elif page == "Performance Dashboard":
    st.header("📊 Performance Dashboard")

    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Model Training' page.")
    else:
        data = load_data()
        history = st.session_state.training_history

        # Training metrics overview
        st.subheader("Training Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_loss = history['train_loss'][-1]
            st.metric("Final Training Loss", f"{final_loss:.4f}")
        with col2:
            best_f1 = max(history['val_f1'])
            st.metric("Best Validation F1", f"{best_f1:.4f}")
        with col3:
            best_precision = max(history['val_precision'])
            st.metric("Best Precision", f"{best_precision:.4f}")
        with col4:
            best_recall = max(history['val_recall'])
            st.metric("Best Recall", f"{best_recall:.4f}")

        # Training curves
        st.subheader("Training Progress")

        col1, col2 = st.columns(2)

        with col1:
            # Loss curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(history['train_loss']))),
                y=history['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title="Training Loss Over Time",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Metrics curves
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(history['val_f1']))),
                y=history['val_f1'],
                mode='lines',
                name='F1 Score',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(history['val_precision']))),
                y=history['val_precision'],
                mode='lines',
                name='Precision',
                line=dict(color='orange')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(history['val_recall']))),
                y=history['val_recall'],
                mode='lines',
                name='Recall',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Validation Metrics Over Time",
                xaxis_title="Epoch",
                yaxis_title="Score",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Model architecture info
        st.subheader("Model Architecture")
        model = st.session_state.model

        col1, col2 = st.columns(2)
        with col1:
            st.info("**GNN Component**")
            st.write(f"- Layers: {model.gnn_model.num_layers}")
            st.write(f"- Hidden Dimension: {model.gnn_model.hidden_dim}")
            st.write(f"- Input Features: {model.gnn_model.input_dim}")

        with col2:
            st.info("**LSTM Component**")
            st.write(f"- Layers: {model.lstm_model.num_layers}")
            st.write(f"- Hidden Dimension: {model.lstm_model.hidden_dim}")
            st.write(f"- Input Features: {model.lstm_model.input_dim}")

        # Dataset statistics
        st.subheader("Dataset Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            fraud_by_type = data.groupby('Transaction_Type')['Fraud_Label'].agg(['count', 'sum']).reset_index()
            fraud_by_type['fraud_rate'] = fraud_by_type['sum'] / fraud_by_type['count'] * 100

            fig = px.bar(fraud_by_type, x='Transaction_Type', y='fraud_rate',
                        title="Fraud Rate by Transaction Type")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fraud_by_location = data.groupby('Location')['Fraud_Label'].agg(['count', 'sum']).reset_index()
            fraud_by_location['fraud_rate'] = fraud_by_location['sum'] / fraud_by_location['count'] * 100

            fig = px.bar(fraud_by_location, x='Location', y='fraud_rate',
                        title="Fraud Rate by Location")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fraud_by_device = data.groupby('Device_Type')['Fraud_Label'].agg(['count', 'sum']).reset_index()
            fraud_by_device['fraud_rate'] = fraud_by_device['sum'] / fraud_by_device['count'] * 100

            fig = px.bar(fraud_by_device, x='Device_Type', y='fraud_rate',
                        title="Fraud Rate by Device Type")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**HNN Fraud Detection System** - Powered by PyTorch Geometric and Streamlit")
