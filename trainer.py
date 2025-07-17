import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

from models.hybrid_model import HybridFraudDetector
from eda.data_preprocessor import DataPreprocessor
from eda.graph_builder import GraphBuilder

class ModelTrainer:
    """Training pipeline for hybrid fraud detection model"""
    
    def __init__(self, hidden_dim=128, gnn_layers=2, lstm_layers=2, dropout=0.2, 
                 learning_rate=0.001, weight_decay=1e-5):
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_accuracy': []
        }
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def _create_data_loaders(self, X, y, batch_size=64, test_size=0.2, val_size=0.1, 
                           use_smote=True, random_state=42):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            X: Feature matrix
            y: Target labels
            batch_size: Batch size for training
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            use_smote: Whether to apply SMOTE for class balancing
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary of data loaders and indices
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        # Apply SMOTE to training data if requested
        if use_smote:
            print("Applying SMOTE for class balancing...")
            preprocessor = DataPreprocessor()
            X_train, y_train = preprocessor.apply_smote(X_train, y_train, random_state=random_state)
            print(f"After SMOTE - Training samples: {len(X_train)}, Fraud rate: {y_train.mean():.3f}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_indices': np.arange(len(X_train)),
            'val_indices': np.arange(len(X_val)),
            'test_indices': np.arange(len(X_test)),
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def _prepare_graph_data(self, data, X, indices, graph_builder):
        """
        Prepare graph data for the given indices
        
        Args:
            data: Original transaction data
            X: Processed features
            indices: Data indices
            graph_builder: Graph builder instance
        
        Returns:
            Graph data
        """
        # Sample data for the given indices
        sampled_data = data.iloc[indices].copy()
        sampled_features = X[indices]
        
        # Build graph
        graph_data = graph_builder.build_graph(sampled_data, sampled_features)
        
        return graph_data
    
    def _prepare_sequence_data(self, X, sequence_length=10):
        """
        Prepare sequence data for LSTM
        
        Args:
            X: Feature matrix
            sequence_length: Length of sequences
        
        Returns:
            Sequence data tensor
        """
        # For simplicity, we'll use sliding window approach
        # In practice, you might want to group by user
        sequences = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
        
        # If we don't have enough data for sequences, duplicate the last features
        if len(sequences) == 0:
            # Create sequences by padding/repeating
            sequences = [np.tile(X, (sequence_length, 1))[:sequence_length] for _ in range(len(X))]
        
        return torch.FloatTensor(np.array(sequences))
    
    def _calculate_class_weights(self, y):
        """Calculate class weights for imbalanced dataset"""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        class_weights = total_samples / (len(unique_classes) * counts)
        weight_dict = dict(zip(unique_classes, class_weights))
        
        weights = torch.FloatTensor([weight_dict[i] for i in range(len(unique_classes))])
        return weights.to(self.device)
    
    def train_epoch(self, model, train_loader, optimizer, criterion, graph_data, epoch):
        """
        Train for one epoch
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss criterion
            graph_data: Graph data
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data(X_batch.cpu().numpy())
            sequence_data = sequence_data.to(self.device)
            
            # Ensure batch size consistency
            batch_size = X_batch.size(0)
            if sequence_data.size(0) != batch_size:
                # Adjust sequence data to match batch size
                if sequence_data.size(0) > batch_size:
                    sequence_data = sequence_data[:batch_size]
                else:
                    # Repeat last sequence to match batch size
                    repeats_needed = batch_size - sequence_data.size(0)
                    last_sequence = sequence_data[-1:].repeat(repeats_needed, 1, 1)
                    sequence_data = torch.cat([sequence_data, last_sequence], dim=0)
            
            # Create batch graph data
            batch_graph = self._create_batch_graph_data(graph_data, batch_size)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_graph, sequence_data)
            loss = criterion(outputs.squeeze(), y_batch.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _create_batch_graph_data(self, graph_data, batch_size):
        """
        Create batch graph data by replicating the graph structure
        
        Args:
            graph_data: Original graph data
            batch_size: Desired batch size
        
        Returns:
            Batched graph data
        """
        # For simplicity, we'll create a batch by replicating the graph
        # In practice, you might want more sophisticated batching
        
        graph_list = []
        for i in range(batch_size):
            # Create a copy of the graph data using our simple structure
            from utils.graph_builder import SimpleGraphData
            batch_graph = SimpleGraphData(
                x=graph_data.x.clone(),
                edge_index=graph_data.edge_index.clone() if graph_data.edge_index is not None else None
            )
            graph_list.append(batch_graph)
        
        # Simple batching - create a simple graph data structure
        if graph_list:
            # For simplicity, use the first graph structure for all batches
            from utils.graph_builder import SimpleGraphData
            batched_graph = SimpleGraphData(
                x=graph_data.x.to(self.device),
                edge_index=graph_data.edge_index.to(self.device) if graph_data.edge_index is not None else None,
                batch=torch.zeros(graph_data.x.size(0), dtype=torch.long).to(self.device)
            )
            return batched_graph
        else:
            from utils.graph_builder import SimpleGraphData
            return SimpleGraphData().to(self.device)
    
    def validate_epoch(self, model, val_loader, criterion, graph_data):
        """
        Validate for one epoch
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss criterion
            graph_data: Graph data
        
        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_probabilities = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Prepare sequence data
                sequence_data = self._prepare_sequence_data(X_batch.cpu().numpy())
                sequence_data = sequence_data.to(self.device)
                
                # Ensure batch size consistency
                batch_size = X_batch.size(0)
                if sequence_data.size(0) != batch_size:
                    if sequence_data.size(0) > batch_size:
                        sequence_data = sequence_data[:batch_size]
                    else:
                        repeats_needed = batch_size - sequence_data.size(0)
                        last_sequence = sequence_data[-1:].repeat(repeats_needed, 1, 1)
                        sequence_data = torch.cat([sequence_data, last_sequence], dim=0)
                
                # Create batch graph data
                batch_graph = self._create_batch_graph_data(graph_data, batch_size)
                
                # Forward pass
                outputs = model(batch_graph, sequence_data)
                loss = criterion(outputs.squeeze(), y_batch.float())
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = (probabilities > 0.5).long()
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                num_batches += 1
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': total_loss / num_batches,
            'f1': f1_score(all_targets, all_predictions, zero_division=0),
            'precision': precision_score(all_targets, all_predictions, zero_division=0),
            'recall': recall_score(all_targets, all_predictions, zero_division=0),
            'accuracy': accuracy_score(all_targets, all_predictions)
        }
        
        return metrics
    
    def train(self, graph_data, X, y, epochs=50, batch_size=64, test_size=0.2, 
              use_smote=True, random_state=42, patience=10):
        """
        Main training loop
        
        Args:
            graph_data: Graph data
            X: Feature matrix
            y: Target labels
            epochs: Number of training epochs
            batch_size: Batch size
            test_size: Test set proportion
            use_smote: Whether to use SMOTE
            random_state: Random state
            patience: Early stopping patience
        
        Returns:
            Trained model and training history
        """
        print("Starting model training...")
        print(f"Dataset size: {len(X)}, Features: {X.shape[1]}")
        print(f"Fraud rate: {y.mean():.3f}")
        
        # Prepare data loaders
        data_splits = self._create_data_loaders(
            X, y, batch_size=batch_size, test_size=test_size, 
            use_smote=use_smote, random_state=random_state
        )
        
        # Initialize model
        gnn_input_dim = graph_data.x.size(1)
        lstm_input_dim = X.shape[1]
        
        model = HybridFraudDetector(
            gnn_input_dim=gnn_input_dim,
            lstm_input_dim=lstm_input_dim,
            hidden_dim=self.hidden_dim,
            gnn_layers=self.gnn_layers,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Calculate class weights for imbalanced dataset
        class_weights = self._calculate_class_weights(data_splits['y_train'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1] / class_weights[0])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(
                model, data_splits['train_loader'], optimizer, criterion, graph_data, epoch
            )
            
            # Validation
            val_metrics = self.validate_epoch(
                model, data_splits['val_loader'], criterion, graph_data
            )
            
            # Update learning rate
            scheduler.step(val_metrics['f1'])
            
            # Store metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['val_precision'].append(val_metrics['precision'])
            self.training_history['val_recall'].append(val_metrics['recall'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}")
                print(f"  Val Precision: {val_metrics['precision']:.4f}")
                print(f"  Val Recall: {val_metrics['recall']:.4f}")
                print()
            
            # Early stopping
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
        print(f"Best validation F1 score: {best_f1:.4f}")
        
        # Final evaluation on test set
        test_metrics = self.validate_epoch(
            model, data_splits['test_loader'], criterion, graph_data
        )
        print("\nTest Set Performance:")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f}")
        print(f"  Test Recall: {test_metrics['recall']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        return model, self.training_history
    
    def save_model(self, model, filepath):
        """Save trained model"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'hidden_dim': self.hidden_dim,
                'gnn_layers': self.gnn_layers,
                'lstm_layers': self.lstm_layers,
                'dropout': self.dropout
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath, gnn_input_dim, lstm_input_dim):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        model = HybridFraudDetector(
            gnn_input_dim=gnn_input_dim,
            lstm_input_dim=lstm_input_dim,
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            gnn_layers=checkpoint['model_config']['gnn_layers'],
            lstm_layers=checkpoint['model_config']['lstm_layers'],
            dropout=checkpoint['model_config']['dropout']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")
        return model

