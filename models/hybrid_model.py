import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_model import GNNModel, GraphSAGEModel, AttentionGNN
from models.lstm_model import LSTMModel, AdvancedLSTM, TemporalConvLSTM

class HybridFraudDetector(nn.Module):
    """Hybrid Neural Network combining GNN and LSTM for fraud detection"""
    
    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim=128, 
                 gnn_layers=2, lstm_layers=2, dropout=0.2, fusion_method='concat'):
        super(HybridFraudDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        self.gnn_layers = gnn_layers
        self.lstm_layers = lstm_layers
        
        # GNN component for relational analysis
        self.gnn_model = GNNModel(
            input_dim=gnn_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )
        
        # LSTM component for temporal analysis
        self.lstm_model = AdvancedLSTM(
            input_dim=lstm_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = hidden_dim * 2
        elif fusion_method == 'add':
            fusion_input_dim = hidden_dim
        elif fusion_method == 'attention':
            fusion_input_dim = hidden_dim
            self.attention_weights = nn.Linear(hidden_dim * 2, 2)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Final classification layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, graph_data, sequence_data):
        """
        Forward pass through hybrid model
        
        Args:
            graph_data: Graph data containing node features, edge indices, and batch info
            sequence_data: Sequential data [batch_size, seq_len, features]
        
        Returns:
            Fraud probability scores [batch_size, 1]
        """
        # GNN forward pass
        gnn_embeddings = self.gnn_model(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            batch=graph_data.batch
        )
        
        # LSTM forward pass
        lstm_embeddings = self.lstm_model(sequence_data)
        
        # Fusion of GNN and LSTM outputs
        if self.fusion_method == 'concat':
            # Concatenate embeddings
            fused_embeddings = torch.cat([gnn_embeddings, lstm_embeddings], dim=1)
        elif self.fusion_method == 'add':
            # Element-wise addition
            fused_embeddings = gnn_embeddings + lstm_embeddings
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            combined = torch.cat([gnn_embeddings, lstm_embeddings], dim=1)
            attention_weights = torch.softmax(self.attention_weights(combined), dim=1)
            fused_embeddings = (attention_weights[:, 0:1] * gnn_embeddings + 
                              attention_weights[:, 1:2] * lstm_embeddings)
        
        # Final classification
        output = self.fusion_layers(fused_embeddings)
        
        return output
    
    def predict(self, graph_data, sequence_data):
        """
        Make predictions with probability scores
        
        Args:
            graph_data: Graph data
            sequence_data: Sequential data
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Fraud probabilities
        """
        with torch.no_grad():
            logits = self.forward(graph_data, sequence_data)
            probabilities = torch.sigmoid(logits).squeeze()
            predictions = (probabilities > 0.5).long()
            
        return predictions, probabilities

class EnsembleHybridModel(nn.Module):
    """Ensemble of multiple hybrid models for improved performance"""
    
    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim=128, 
                 num_models=3, gnn_layers=2, lstm_layers=2, dropout=0.2):
        super(EnsembleHybridModel, self).__init__()
        
        self.num_models = num_models
        
        # Create ensemble of hybrid models
        self.models = nn.ModuleList([
            HybridFraudDetector(
                gnn_input_dim=gnn_input_dim,
                lstm_input_dim=lstm_input_dim,
                hidden_dim=hidden_dim,
                gnn_layers=gnn_layers,
                lstm_layers=lstm_layers,
                dropout=dropout,
                fusion_method='concat' if i == 0 else ('add' if i == 1 else 'attention')
            )
            for i in range(num_models)
        ])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, graph_data, sequence_data):
        """
        Forward pass through ensemble
        
        Args:
            graph_data: Graph data
            sequence_data: Sequential data
        
        Returns:
            Ensemble predictions
        """
        outputs = []
        
        for model in self.models:
            output = model(graph_data, sequence_data)
            outputs.append(output)
        
        # Weighted ensemble
        outputs = torch.stack(outputs, dim=0)  # [num_models, batch_size, 1]
        weights = torch.softmax(self.ensemble_weights, dim=0)
        ensemble_output = torch.sum(outputs * weights.view(-1, 1, 1), dim=0)
        
        return ensemble_output

class AdaptiveHybridModel(nn.Module):
    """Adaptive hybrid model that learns to weight GNN vs LSTM based on input"""
    
    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim=128, 
                 gnn_layers=2, lstm_layers=2, dropout=0.2):
        super(AdaptiveHybridModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # GNN component
        self.gnn_model = AttentionGNN(
            input_dim=gnn_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )
        
        # LSTM component
        self.lstm_model = TemporalConvLSTM(
            input_dim=lstm_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # Adaptive weighting network
        self.adaptive_weights = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, graph_data, sequence_data):
        """
        Forward pass with adaptive weighting
        
        Args:
            graph_data: Graph data
            sequence_data: Sequential data
        
        Returns:
            Fraud predictions
        """
        # Get embeddings from both models
        gnn_embeddings = self.gnn_model(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            batch=graph_data.batch
        )
        lstm_embeddings = self.lstm_model(sequence_data)
        
        # Compute adaptive weights
        combined_features = torch.cat([gnn_embeddings, lstm_embeddings], dim=1)
        weights = self.adaptive_weights(combined_features)
        
        # Adaptive fusion
        fused_embeddings = (weights[:, 0:1] * gnn_embeddings + 
                           weights[:, 1:2] * lstm_embeddings)
        
        # Final classification
        output = self.classifier(fused_embeddings)
        
        return output, weights

class MultitaskHybridModel(nn.Module):
    """Multitask learning model for fraud detection and risk assessment"""
    
    def __init__(self, gnn_input_dim, lstm_input_dim, hidden_dim=128, 
                 gnn_layers=2, lstm_layers=2, dropout=0.2):
        super(MultitaskHybridModel, self).__init__()
        
        # Shared backbone
        self.gnn_model = GNNModel(
            input_dim=gnn_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )
        
        self.lstm_model = LSTMModel(
            input_dim=lstm_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Task-specific heads
        # Fraud detection head
        self.fraud_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Risk score prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Risk score between 0 and 1
        )
        
        # Transaction amount prediction head (auxiliary task)
        self.amount_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, graph_data, sequence_data):
        """
        Forward pass for multitask learning
        
        Args:
            graph_data: Graph data
            sequence_data: Sequential data
        
        Returns:
            Dictionary of task predictions
        """
        # Get embeddings
        gnn_embeddings = self.gnn_model(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            batch=graph_data.batch
        )
        lstm_embeddings = self.lstm_model(sequence_data)
        
        # Shared representation
        combined_embeddings = torch.cat([gnn_embeddings, lstm_embeddings], dim=1)
        shared_repr = self.shared_layer(combined_embeddings)
        
        # Task-specific predictions
        fraud_pred = self.fraud_head(shared_repr)
        risk_pred = self.risk_head(shared_repr)
        amount_pred = self.amount_head(shared_repr)
        
        return {
            'fraud': fraud_pred,
            'risk': risk_pred,
            'amount': amount_pred
        }
