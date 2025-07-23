import streamlit as st
import pandas as pd
import numpy as np
import torch
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
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Data Overview", "Model Training", "Model Evaluation", "Real-time Prediction", "Performance Dashboard"]
)

def load_data():
    """Load and reduce the dataset to a maximum of 10,000 rows"""
    if 'data' not in st.session_state:
        try:
            # Load the full dataset
            data = pd.read_csv('dataset/synthetic_fraud_dataset.csv')

            # Limit to 10,000 rows if larger
            if len(data) > 10000:
                data = data.sample(n=10000, random_state=42).reset_index(drop=True)

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
        epochs = st.slider("Number of Epochs", 5, 100, 50)
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    
    with col2:
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
        hidden_dim = st.slider("Hidden Dimension", 32, 256, 128)
    
    with col3:
        gnn_layers = st.slider("GNN Layers", 1, 5, 2)
        lstm_layers = st.slider("LSTM Layers", 1, 3, 2)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        use_smote = st.checkbox("Use SMOTE for Class Balancing", value=True)
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
                    fig = px.line(x=range(len(history['train_loss'])), y=history['train_loss'],
                                 title="Training Loss Over Time")
                    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Validation Metrics")
                    metrics_df = pd.DataFrame({
                        'Epoch': range(len(history['val_f1'])),
                        'F1 Score': history['val_f1'],
                        'Precision': history['val_precision'],
                        'Recall': history['val_recall']
                    })
                    
                    fig = px.line(metrics_df, x='Epoch', y=['F1 Score', 'Precision', 'Recall'],
                                 title="Validation Metrics Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                
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
                
                # Make predictions
                predictions, probabilities = evaluator.evaluate_model(
                    model, graph_data, X_processed, y
                )
                
                # Calculate metrics
                f1 = f1_score(y, predictions)
                auc = roc_auc_score(y, probabilities)
                cm = confusion_matrix(y, predictions)
                report = classification_report(y, predictions, output_dict=True)
                
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
                    fpr, tpr, _ = evaluator.calculate_roc_curve(y, probabilities)
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
            risk_score = st.slider("Risk Score", 0.0, 1.0, 0.5)
        
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
                    'Risk_Score': risk_score,
                    'IP_Address_Flag': int(ip_flag),
                    'Previous_Fraudulent_Activity': int(prev_fraud),
                    'Is_Weekend': int(is_weekend),
                    'Avg_Transaction_Amount_7d': avg_amount_7d,
                    'Failed_Transaction_Count_7d': failed_count_7d,
                    'Transaction_Distance': transaction_distance,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
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
                    lstm_out = model.lstm_model(seq_features)
                    prediction_prob = torch.sigmoid(lstm_out).item()
                    prediction = 1 if prediction_prob > 0.5 else 0
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 FRAUD DETECTED")
                        st.error(f"Fraud Probability: {prediction_prob:.4f}")
                    else:
                        st.success("✅ LEGITIMATE TRANSACTION")
                        st.success(f"Fraud Probability: {prediction_prob:.4f}")
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prediction_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability"},
                        delta = {'reference': 0.5},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "red" if prediction_prob > 0.5 else "green"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5}}))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors analysis
                st.subheader("Risk Factor Analysis")
                risk_factors = []
                
                if amount > 1000:
                    risk_factors.append("High transaction amount")
                if risk_score > 0.7:
                    risk_factors.append("High risk score")
                if ip_flag:
                    risk_factors.append("Suspicious IP address")
                if prev_fraud:
                    risk_factors.append("Previous fraudulent activity")
                if failed_count_7d > 5:
                    risk_factors.append("High number of recent failed transactions")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.info("No significant risk factors detected")
                
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
