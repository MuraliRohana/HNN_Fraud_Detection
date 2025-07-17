import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """LSTM model for temporal pattern analysis in fraud detection"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(lstm_output_dim)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through LSTM
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
        
        Returns:
            Output embeddings [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Apply batch normalization and dropout
        context_vector = self.batch_norm(context_vector)
        context_vector = self.dropout_layer(context_vector)
        
        # Final layers
        out = F.relu(self.fc1(context_vector))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out

class GRUModel(nn.Module):
    """GRU model for temporal pattern analysis"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, bidirectional=False):
        super(GRUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate GRU output dimension
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.Linear(gru_output_dim, 1)
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(gru_output_dim)
        self.fc1 = nn.Linear(gru_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through GRU
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
        
        Returns:
            Output embeddings [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(gru_out * attention_weights, dim=1)
        
        # Apply batch normalization and dropout
        context_vector = self.batch_norm(context_vector)
        context_vector = self.dropout_layer(context_vector)
        
        # Final layers
        out = F.relu(self.fc1(context_vector))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out

class AdvancedLSTM(nn.Module):
    """Advanced LSTM with multiple attention mechanisms"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, 
                 bidirectional=True, use_self_attention=True):
        super(AdvancedLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_self_attention = use_self_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Self-attention mechanism
        if use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout
            )
        
        # Global attention for sequence pooling
        self.global_attention = nn.Linear(lstm_output_dim, 1)
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Multi-layer perceptron for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through Advanced LSTM
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
        
        Returns:
            Output embeddings [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention
        if self.use_self_attention:
            # Transpose for attention (seq_len, batch_size, hidden_dim)
            lstm_out_transposed = lstm_out.transpose(0, 1)
            attended_out, _ = self.self_attention(
                lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
            )
            # Transpose back
            lstm_out = attended_out.transpose(0, 1)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Global attention pooling
        attention_weights = torch.softmax(self.global_attention(lstm_out), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Apply dropout
        context_vector = self.dropout_layer(context_vector)
        
        # Final MLP
        output = self.mlp(context_vector)
        
        return output

class TemporalConvLSTM(nn.Module):
    """Combination of 1D CNN and LSTM for temporal fraud detection"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(TemporalConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 1D Convolutional layers for local pattern extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention and output layers
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        """
        Forward pass through Temporal Conv-LSTM
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
        
        Returns:
            Output embeddings [batch_size, output_dim]
        """
        # Transpose for conv1d (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Apply convolutional layers
        conv_out = self.conv_layers(x_conv)
        
        # Transpose back for LSTM (batch_size, seq_len, hidden_dim)
        conv_out = conv_out.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(conv_out)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Apply dropout and output layer
        context_vector = self.dropout_layer(context_vector)
        output = self.output_layer(context_vector)
        
        return output
