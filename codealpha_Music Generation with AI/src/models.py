"""
Neural Network Models for Music Generation
Includes LSTM-based architecture for sequence prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MusicLSTM(nn.Module):
    """
    LSTM-based model for music generation
    Uses embedding layer, stacked LSTM, and fully connected output
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.3):
        """
        Initialize the Music LSTM model
        
        Args:
            vocab_size: Number of unique notes/chords in vocabulary
            embedding_dim: Dimension of note embeddings
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(MusicLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional tuple of (h_n, c_n) hidden states
            
        Returns:
            Tuple of (output logits, hidden states)
        """
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch, seq_len, hidden_dim)
        
        # Take only the last output for prediction
        lstm_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Dropout and fully connected
        out = self.dropout(lstm_out)
        out = self.fc(out)  # (batch, vocab_size)
        
        return out, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)


class MusicTransformer(nn.Module):
    """
    Transformer-based model for music generation
    Uses self-attention mechanism for capturing long-range dependencies
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024, 
                 dropout: float = 0.1, max_seq_length: int = 512):
        """
        Initialize the Music Transformer model
        
        Args:
            vocab_size: Number of unique notes/chords in vocabulary
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(MusicTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            mask: Optional attention mask
            
        Returns:
            Output logits of shape (batch_size, vocab_size)
        """
        # Embedding with scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create causal mask for autoregressive generation
        if mask is None:
            seq_len = x.size(1)
            mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask)
        
        # Take the last position output
        x = x[:, -1, :]
        
        # Output projection
        out = self.fc(x)
        
        return out
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for transformer"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MusicVAE(nn.Module):
    """
    Variational Autoencoder for music generation
    Learns a latent space representation of music sequences
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, latent_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize Music VAE
        
        Args:
            vocab_size: Number of unique notes/chords
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of encoder/decoder
            latent_dim: Dimension of latent space
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(MusicVAE, self).__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent space"""
        embedded = self.embedding(x)
        _, (h_n, _) = self.encoder(embedded)
        h_n = h_n[-1]  # Take last layer hidden state
        
        mu = self.fc_mu(h_n)
        log_var = self.fc_var(h_n)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Decode from latent space"""
        batch_size = z.size(0)
        
        # Initialize decoder hidden state from latent
        hidden = self.latent_to_hidden(z)
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        
        # Start token (use 0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        
        outputs = []
        for _ in range(seq_length):
            embedded = self.embedding(decoder_input)
            output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            output = self.fc_out(output)
            outputs.append(output)
            
            # Use argmax for next input (teacher forcing can be added)
            decoder_input = output.argmax(dim=-1)
            
        return torch.cat(outputs, dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z, x.size(1))
        
        return output, mu, log_var
    
    def generate(self, num_samples: int, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate new sequences from random latent vectors"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, seq_length)


def get_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
    """
    Factory function to get model by type
    
    Args:
        model_type: 'lstm', 'transformer', or 'vae'
        vocab_size: Vocabulary size
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    models = {
        'lstm': MusicLSTM,
        'transformer': MusicTransformer,
        'vae': MusicVAE
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
        
    return models[model_type](vocab_size, **kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...")
    
    vocab_size = 100
    batch_size = 4
    seq_length = 50
    
    # Test input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Test LSTM
    print("\nTesting MusicLSTM...")
    lstm_model = MusicLSTM(vocab_size)
    output, hidden = lstm_model(x)
    print(f"LSTM output shape: {output.shape}")
    
    # Test Transformer
    print("\nTesting MusicTransformer...")
    transformer_model = MusicTransformer(vocab_size)
    output = transformer_model(x)
    print(f"Transformer output shape: {output.shape}")
    
    # Test VAE
    print("\nTesting MusicVAE...")
    vae_model = MusicVAE(vocab_size)
    output, mu, log_var = vae_model(x)
    print(f"VAE output shape: {output.shape}")
    print(f"VAE latent mu shape: {mu.shape}")
    
    print("\nAll models working correctly!")
