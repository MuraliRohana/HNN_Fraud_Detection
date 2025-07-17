import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    """Simplified Graph Neural Network model for analyzing transaction relationships"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Use standard MLP layers as a simplified GNN
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final projection layer
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass through the simplified GNN
        
        Args:
            x: Node features [num_nodes, input_dim] or batch features [batch_size, input_dim]
            edge_index: Edge connectivity (not used in simplified version)
            batch: Batch vector for graph-level prediction
        
        Returns:
            Graph-level embeddings [batch_size, output_dim]
        """
        # Apply MLP layers (simplified graph processing)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Global pooling for graph-level representation (simplified)
        if batch is not None and len(x.shape) > 2:
            # Mean pooling across nodes
            x = torch.mean(x, dim=1)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x

class GraphSAGEModel(nn.Module):
    """Simplified GraphSAGE-inspired model"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Use MLP layers to simulate GraphSAGE
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass through simplified GraphSAGE
        
        Args:
            x: Node features [num_nodes, input_dim] or batch features [batch_size, input_dim]
            edge_index: Edge connectivity (not used in simplified version)
            batch: Batch vector for graph-level prediction
        
        Returns:
            Graph-level embeddings [batch_size, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        x = self.projection(x)
        return x

class AttentionGNN(nn.Module):
    """Simplified GNN with attention mechanism for fraud detection"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, num_heads=4):
        super(AttentionGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # MLP layers with attention
        self.layers = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.attentions.append(nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final layers
        self.global_attention = nn.Linear(hidden_dim, 1)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass with attention
        
        Args:
            x: Node features [batch_size, input_dim] or [num_nodes, input_dim]
            edge_index: Edge connectivity (not used in simplified version)
            batch: Batch vector for graph-level prediction
        
        Returns:
            Graph-level embeddings [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Apply MLP layers with self-attention
        for i, (layer, attention) in enumerate(zip(self.layers, self.attentions)):
            # MLP layer
            x_mlp = layer(x)
            x_mlp = F.relu(x_mlp)
            
            # Self-attention (if batch size > 1)
            if x.dim() == 2 and x.size(0) > 1:
                x_input = x.unsqueeze(0) if x.dim() == 2 else x
                x_att, _ = attention(x_input, x_input, x_input)
                x_att = x_att.squeeze(0) if x_att.dim() == 3 and x_att.size(0) == 1 else x_att
            else:
                x_att = x
            
            # Combine and apply dropout
            x = x_mlp + x_att
            x = self.dropout_layer(x)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x
