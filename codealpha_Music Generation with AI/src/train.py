"""
Training script for Music Generation models
Supports LSTM, Transformer, and VAE architectures
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MusicLSTM, MusicTransformer, MusicVAE, get_model
from src.data_processing import MidiProcessor, create_sample_data
from config import MODEL_CONFIG, DATA_CONFIG


class MusicTrainer:
    """Trainer class for music generation models"""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.001):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, MusicVAE):
                outputs, mu, log_var = self.model(inputs)
                # VAE loss = reconstruction + KL divergence
                recon_loss = self.criterion(outputs.view(-1, outputs.size(-1)), 
                                           inputs.view(-1))
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + 0.001 * kl_loss  # Beta-VAE with small beta
            else:
                if isinstance(self.model, MusicLSTM):
                    outputs, _ = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if isinstance(self.model, MusicVAE):
                    outputs, mu, log_var = self.model(inputs)
                    recon_loss = self.criterion(outputs.view(-1, outputs.size(-1)), 
                                               inputs.view(-1))
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + 0.001 * kl_loss
                else:
                    if isinstance(self.model, MusicLSTM):
                        outputs, _ = self.model(inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 100, save_path: str = "model.pth", 
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 50)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                current_loss = val_loss
            else:
                val_loss = None
                current_loss = train_loss
                
            # Learning rate scheduling
            self.scheduler.step(current_loss)
            
            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                self.save_model(save_path)
                print(f"  → New best model saved! Loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
                
            # Print epoch summary
            elapsed = time.time() - start_time
            val_str = f", Val Loss: {val_loss:.4f}" if val_loss else ""
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}{val_str} "
                  f"- Time: {elapsed:.1f}s")
            print("-" * 50)
            
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        
    def load_model(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})


def prepare_data(midi_folder: str, sequence_length: int, 
                 batch_size: int, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, MidiProcessor]:
    """
    Prepare data loaders for training
    
    Args:
        midi_folder: Folder containing MIDI files
        sequence_length: Length of input sequences
        batch_size: Batch size
        val_split: Validation split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, processor)
    """
    processor = MidiProcessor(sequence_length=sequence_length)
    
    # Parse MIDI files
    notes = processor.parse_midi_folder(midi_folder)
    
    if len(notes) < sequence_length + 1:
        raise ValueError(f"Not enough notes ({len(notes)}) for sequence length {sequence_length}")
    
    # Prepare sequences
    network_input, network_output = processor.prepare_sequences(notes)
    
    # Convert to tensors
    X = torch.LongTensor(network_input)
    y = torch.LongTensor(network_output)
    
    # Split into train/val
    n_samples = len(X)
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, processor


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Music Generation Model")
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'transformer', 'vae'],
                       help='Model architecture to use')
    parser.add_argument('--midi_folder', type=str, default='data/midi_files',
                       help='Folder containing MIDI files')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=MODEL_CONFIG['batch_size'],
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=MODEL_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=MODEL_CONFIG['sequence_length'],
                       help='Input sequence length')
    parser.add_argument('--create_sample_data', action='store_true',
                       help='Create sample MIDI data for testing')
    parser.add_argument('--output_model', type=str, default=DATA_CONFIG['model_save_path'],
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample MIDI data...")
        create_sample_data(args.midi_folder, num_samples=10)
    
    # Check if MIDI folder exists
    if not os.path.exists(args.midi_folder):
        print(f"MIDI folder not found: {args.midi_folder}")
        print("Creating sample data...")
        create_sample_data(args.midi_folder, num_samples=10)
    
    # Prepare data
    print("\nPreparing data...")
    try:
        train_loader, val_loader, processor = prepare_data(
            args.midi_folder,
            args.sequence_length,
            args.batch_size
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install music21: pip install music21")
        return
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    # Save processor vocabulary
    vocab_path = os.path.join(os.path.dirname(args.output_model), 'vocabulary.pkl')
    processor.save_processed_data(
        vocab_path, 
        [], 
        np.array([]), 
        np.array([])
    )
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model = get_model(
        args.model,
        vocab_size=processor.n_vocab,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    # Create trainer
    trainer = MusicTrainer(model, device, args.learning_rate)
    
    # Train
    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)
    
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        save_path=args.output_model
    )
    
    # Save training history
    history_path = os.path.join(os.path.dirname(args.output_model), 'training_history.pkl')
    import pickle
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to: {history_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
