import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:

    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
        
        # Define categorical and numerical columns
        self.categorical_columns = [
            'Transaction_Type', 'Device_Type', 'Location', 
            'Merchant_Category', 'Card_Type', 'Authentication_Method'
        ]
        
        self.numerical_columns = [
            'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
            'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d', 'Card_Age',
            'Transaction_Distance', 'Risk_Score'
        ]
        
        self.binary_columns = [
            'IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend'
        ]
    
    def _parse_timestamp(self, data):
        data = data.copy()
        
        # Parse timestamp
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Extract temporal features
        data['Hour'] = data['Timestamp'].dt.hour
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
        data['Month'] = data['Timestamp'].dt.month
        data['IsNight'] = ((data['Hour'] >= 22) | (data['Hour'] <= 6)).astype(int)
        data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
        
        # Update Is_Weekend if not present
        if 'Is_Weekend' not in data.columns:
            data['Is_Weekend'] = data['IsWeekend']
        
        return data
    
    def _create_derived_features(self, data):
        data = data.copy()
        
        # Amount-based features
        data['Amount_Log'] = np.log1p(data['Transaction_Amount'])
        data['Amount_Zscore'] = (data['Transaction_Amount'] - data['Avg_Transaction_Amount_7d']) / (data['Avg_Transaction_Amount_7d'] + 1e-8)
        data['Amount_Ratio_Balance'] = data['Transaction_Amount'] / (data['Account_Balance'] + 1e-8)
        
        # Transaction frequency features
        data['Transaction_Velocity'] = data['Daily_Transaction_Count'] / 24.0  # per hour
        data['Failure_Rate'] = data['Failed_Transaction_Count_7d'] / (data['Daily_Transaction_Count'] * 7 + 1e-8)
        
        # Risk-based features
        data['High_Risk'] = (data['Risk_Score'] > 0.7).astype(int)
        data['Risk_Amount_Interaction'] = data['Risk_Score'] * data['Transaction_Amount']
        
        # Distance-based features
        data['Distance_Log'] = np.log1p(data['Transaction_Distance'])
        data['High_Distance'] = (data['Transaction_Distance'] > 1000).astype(int)
        
        # Card age features
        data['New_Card'] = (data['Card_Age'] < 30).astype(int)
        data['Old_Card'] = (data['Card_Age'] > 365).astype(int)
        
        return data
    
    def _encode_categorical_features(self, data, is_training=True):
        data = data.copy()
        
        for col in self.categorical_columns:
            if col in data.columns:
                if is_training:
                    # Fit encoder
                    encoder = LabelEncoder()
                    data[col + '_encoded'] = encoder.fit_transform(data[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    # Transform using fitted encoder
                    encoder = self.encoders[col]
                    # Handle unseen categories
                    unique_values = set(encoder.classes_)
                    data[col] = data[col].astype(str).apply(
                        lambda x: x if x in unique_values else encoder.classes_[0]
                    )
                    data[col + '_encoded'] = encoder.transform(data[col])
        
        return data
    
    def _scale_numerical_features(self, data, is_training=True):
        data = data.copy()
        
        # Get all numerical columns including derived ones
        numerical_cols = []
        for col in data.columns:
            if (col in self.numerical_columns or 
                col.endswith('_Log') or 
                col.endswith('_Zscore') or 
                col.endswith('_Ratio_Balance') or
                col in ['Transaction_Velocity', 'Failure_Rate', 'Risk_Amount_Interaction']):
                numerical_cols.append(col)
        
        for col in numerical_cols:
            if col in data.columns:
                if is_training:
                    # Fit scaler
                    scaler = RobustScaler()
                    data[col + '_scaled'] = scaler.fit_transform(data[[col]])
                    self.scalers[col] = scaler
                else:
                    # Transform using fitted scaler
                    scaler = self.scalers[col]
                    data[col + '_scaled'] = scaler.transform(data[[col]])
        
        return data
    
    def _select_features(self, X, y, k=50):
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = self.feature_selector.get_support(indices=True)
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def _prepare_features(self, data):
        feature_columns = []
        
        # Add encoded categorical features
        for col in self.categorical_columns:
            if col + '_encoded' in data.columns:
                feature_columns.append(col + '_encoded')
        
        # Add scaled numerical features
        for col in data.columns:
            if col.endswith('_scaled'):
                feature_columns.append(col)
        
        # Add binary features
        for col in self.binary_columns:
            if col in data.columns:
                feature_columns.append(col)
        
        # Add derived binary features
        derived_binary = ['IsNight', 'High_Risk', 'High_Distance', 'New_Card', 'Old_Card']
        for col in derived_binary:
            if col in data.columns:
                feature_columns.append(col)
        
        # Add temporal features
        temporal_features = ['Hour', 'DayOfWeek', 'Month']
        for col in temporal_features:
            if col in data.columns:
                feature_columns.append(col)
        
        return data[feature_columns].values
    
    def preprocess(self, data, is_prediction=False):

        # Create a copy to avoid modifying original data
        data_processed = data.copy()
        
        # Parse timestamp and create temporal features
        data_processed = self._parse_timestamp(data_processed)
        
        # Create derived features
        data_processed = self._create_derived_features(data_processed)
        
        # Handle missing values
        data_processed = data_processed.fillna(data_processed.median(numeric_only=True))
        
        # Encode categorical features
        is_training = not self.is_fitted and not is_prediction
        data_processed = self._encode_categorical_features(data_processed, is_training)
        
        # Scale numerical features
        data_processed = self._scale_numerical_features(data_processed, is_training)
        
        # Prepare feature matrix
        X = self._prepare_features(data_processed)
        
        # Feature selection (only during training)
        if not is_prediction:
            y = data['Fraud_Label'].values if 'Fraud_Label' in data.columns else None
            if is_training and y is not None:
                X = self._select_features(X, y)
                self.is_fitted = True
            elif self.feature_selector is not None:
                X = X[:, self.selected_features]
            
            return X, y
        else:
            if self.feature_selector is not None:
                X = X[:, self.selected_features]
            return X, None
    
    def create_sequences(self, X, y=None, sequence_length=10, user_col=None):

        if user_col is not None:
            # Group by user and create sequences
            unique_users = np.unique(user_col)
            X_sequences = []
            y_sequences = [] if y is not None else None
            
            for user in unique_users:
                user_mask = (user_col == user)
                user_X = X[user_mask]
                user_y = y[user_mask] if y is not None else None
                
                # Create sequences for this user
                for i in range(len(user_X) - sequence_length + 1):
                    X_sequences.append(user_X[i:i+sequence_length])
                    if y is not None:
                        y_sequences.append(user_y[i+sequence_length-1])  # Predict last label
            
            X_sequences = np.array(X_sequences)
            if y is not None:
                y_sequences = np.array(y_sequences)
        else:
            # Simple sliding window approach
            X_sequences = []
            y_sequences = [] if y is not None else None
            
            for i in range(len(X) - sequence_length + 1):
                X_sequences.append(X[i:i+sequence_length])
                if y is not None:
                    y_sequences.append(y[i+sequence_length-1])
            
            X_sequences = np.array(X_sequences)
            if y is not None:
                y_sequences = np.array(y_sequences)
        
        if y is not None:
            return X_sequences, y_sequences
        else:
            return X_sequences
    
    def apply_smote(self, X, y, random_state=42):

        # Create pipeline with SMOTE and undersampling
        over = SMOTE(sampling_strategy=0.3, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
        
        steps = [('over', over), ('under', under)]
        pipeline = ImbPipeline(steps=steps)
        
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        
        return X_resampled, y_resampled
    
    def get_feature_names(self):
        if not self.is_fitted:
            return None
        
        # Reconstruct feature names
        feature_names = []
        
        # Categorical features
        for col in self.categorical_columns:
            feature_names.append(col + '_encoded')
        
        # Numerical features (scaled)
        for col in self.numerical_columns:
            feature_names.append(col + '_scaled')
        
        # Derived numerical features
        derived_numerical = [
            'Amount_Log_scaled', 'Amount_Zscore_scaled', 'Amount_Ratio_Balance_scaled',
            'Transaction_Velocity_scaled', 'Failure_Rate_scaled', 'Risk_Amount_Interaction_scaled',
            'Distance_Log_scaled'
        ]
        feature_names.extend(derived_numerical)
        
        # Binary features
        feature_names.extend(self.binary_columns)
        feature_names.extend(['IsNight', 'High_Risk', 'High_Distance', 'New_Card', 'Old_Card'])
        
        # Temporal features
        feature_names.extend(['Hour', 'DayOfWeek', 'Month'])
        
        if self.selected_features is not None:
            return [feature_names[i] for i in self.selected_features]
        
        return feature_names
