"""
Music Generation Script
Generate new music using trained models
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MusicLSTM, MusicTransformer, MusicVAE, get_model
from src.data_processing import MidiProcessor, MidiGenerator
from config import MODEL_CONFIG, DATA_CONFIG, GENERATION_CONFIG, MUSIC_CONFIG


class MusicGenerator:
    """Generate music using trained models"""
    
    def __init__(self, model_path: str, vocab_path: str, model_type: str = 'lstm',
                 device: Optional[torch.device] = None):
        """
        Initialize music generator
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary pickle file
            model_type: Type of model ('lstm', 'transformer', 'vae')
            device: Device to use (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Load vocabulary
        self.load_vocabulary(vocab_path)
        
        # Load model
        self.load_model(model_path)
        
        # MIDI generator
        self.midi_generator = MidiGenerator(
            tempo=MUSIC_CONFIG['tempo'],
            time_signature=MUSIC_CONFIG['time_signature']
        )
        
    def load_vocabulary(self, vocab_path: str) -> None:
        """Load vocabulary mappings"""
        with open(vocab_path, 'rb') as f:
            data = pickle.load(f)
            
        self.note_to_int = data['note_to_int']
        self.int_to_note = data['int_to_note']
        self.n_vocab = data['n_vocab']
        self.sequence_length = data['sequence_length']
        
        print(f"Loaded vocabulary with {self.n_vocab} unique notes/chords")
        
    def load_model(self, model_path: str) -> None:
        """Load trained model"""
        # Create model architecture
        self.model = get_model(
            self.model_type,
            vocab_size=self.n_vocab,
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        
    def sample_with_temperature(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """
        Sample from logits with temperature scaling
        
        Args:
            logits: Model output logits
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Sampled index
        """
        if temperature == 0:
            # Greedy sampling
            return torch.argmax(logits).item()
            
        # Apply temperature
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sample from distribution
        return torch.multinomial(probs, 1).item()
    
    def generate(self, seed_sequence: Optional[List[int]] = None, 
                length: int = 500, temperature: float = 1.0,
                top_k: Optional[int] = None, top_p: Optional[float] = None) -> List[str]:
        """
        Generate music sequence
        
        Args:
            seed_sequence: Starting sequence (random if None)
            length: Number of notes to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling threshold (optional)
            
        Returns:
            List of generated note strings
        """
        self.model.eval()
        
        # Initialize seed sequence
        if seed_sequence is None:
            # Random seed from vocabulary
            seed_sequence = [np.random.randint(0, self.n_vocab) 
                           for _ in range(self.sequence_length)]
        elif len(seed_sequence) < self.sequence_length:
            # Pad with random notes
            padding = [np.random.randint(0, self.n_vocab) 
                      for _ in range(self.sequence_length - len(seed_sequence))]
            seed_sequence = padding + seed_sequence
            
        # Convert to tensor
        pattern = seed_sequence[-self.sequence_length:]
        generated = list(pattern)
        
        print(f"Generating {length} notes with temperature {temperature}...")
        
        with torch.no_grad():
            hidden = None
            
            for i in range(length):
                # Prepare input
                input_seq = torch.LongTensor([pattern]).to(self.device)
                
                # Forward pass
                if self.model_type == 'vae':
                    # For VAE, we need special handling
                    output, _, _ = self.model(input_seq)
                    logits = output[0, -1, :]
                elif self.model_type == 'lstm':
                    output, hidden = self.model(input_seq, hidden)
                    logits = output[0]
                else:  # transformer
                    output = self.model(input_seq)
                    logits = output[0]
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(0, top_k_indices, top_k_values)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next note
                next_note = self.sample_with_temperature(logits, temperature)
                
                # Update pattern for next iteration
                pattern = pattern[1:] + [next_note]
                generated.append(next_note)
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{length} notes")
        
        # Convert integers to note strings
        generated_notes = [self.int_to_note[idx] for idx in generated[self.sequence_length:]]
        
        print(f"Generation complete! {len(generated_notes)} notes generated")
        return generated_notes
    
    def generate_and_save(self, output_path: str, length: int = 500,
                         temperature: float = 1.0, seed_sequence: Optional[List[int]] = None,
                         **kwargs) -> str:
        """
        Generate music and save to MIDI file
        
        Args:
            output_path: Path to save MIDI file
            length: Number of notes to generate
            temperature: Sampling temperature
            seed_sequence: Optional seed sequence
            **kwargs: Additional generation parameters
            
        Returns:
            Path to saved file
        """
        # Generate notes
        generated_notes = self.generate(
            seed_sequence=seed_sequence,
            length=length,
            temperature=temperature,
            **kwargs
        )
        
        # Save to MIDI
        self.midi_generator.create_midi_with_music21(generated_notes, output_path)
        
        return output_path


def create_demo_generation(output_folder: str) -> None:
    """
    Create a demo generation without trained model
    Uses simple patterns for demonstration
    """
    from src.data_processing import MidiGenerator
    
    print("Creating demo music generation...")
    
    # Musical patterns
    patterns = {
        'c_major_scale': [60, 62, 64, 65, 67, 69, 71, 72],
        'c_major_arpeggio': [60, 64, 67, 72, 67, 64, 60],
        'g_major_scale': [67, 69, 71, 72, 74, 76, 78, 79],
        'a_minor': [69, 71, 72, 74, 76, 77, 79, 81],
    }
    
    # Generate demo sequences
    generator = MidiGenerator(tempo=120)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Simple scale demo
    scale_sequence = patterns['c_major_scale'] * 4
    generator.create_midi_with_midiutil(
        scale_sequence,
        os.path.join(output_folder, 'demo_scale.mid')
    )
    
    # 2. Random pattern combination
    combined = []
    for _ in range(8):
        pattern_name = np.random.choice(list(patterns.keys()))
        pattern = patterns[pattern_name]
        transposition = np.random.choice([0, -12, 12])
        combined.extend([n + transposition for n in pattern])
    
    generator.create_midi_with_midiutil(
        combined,
        os.path.join(output_folder, 'demo_random.mid')
    )
    
    # 3. Melodic sequence with variation
    melody = []
    base_notes = [60, 64, 67, 72]  # C major chord
    for _ in range(16):
        note = np.random.choice(base_notes)
        # Add passing tones
        melody.append(note)
        if np.random.random() > 0.5:
            melody.append(note + np.random.choice([-1, 1, 2]))
    
    generator.create_midi_with_midiutil(
        melody,
        os.path.join(output_folder, 'demo_melody.mid')
    )
    
    print(f"Demo files created in {output_folder}")


def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description="Generate Music with AI")
    parser.add_argument('--model_path', type=str, default=DATA_CONFIG['model_save_path'],
                       help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, default='models/vocabulary.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'vae'],
                       help='Model architecture')
    parser.add_argument('--output', type=str, default=None,
                       help='Output MIDI file path')
    parser.add_argument('--length', type=int, default=GENERATION_CONFIG['default_length'],
                       help='Number of notes to generate')
    parser.add_argument('--temperature', type=float, default=MODEL_CONFIG['temperature'],
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Nucleus sampling threshold')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo generation without trained model')
    
    args = parser.parse_args()
    
    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(
            DATA_CONFIG['generated_output_folder'],
            f"generated_{timestamp}.mid"
        )
    
    # Demo mode
    if args.demo:
        create_demo_generation(DATA_CONFIG['generated_output_folder'])
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train a model first using: python src/train.py")
        print("\nRunning demo generation instead...")
        create_demo_generation(DATA_CONFIG['generated_output_folder'])
        return
    
    if not os.path.exists(args.vocab_path):
        print(f"Vocabulary not found at {args.vocab_path}")
        return
    
    # Initialize generator
    generator = MusicGenerator(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        model_type=args.model_type
    )
    
    # Generate and save
    output_path = generator.generate_and_save(
        output_path=args.output,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print(f"\nGenerated music saved to: {output_path}")
    print("\nTo play the MIDI file, you can:")
    print("  - Open it in a music player that supports MIDI")
    print("  - Use a DAW (Digital Audio Workstation)")
    print("  - Convert to audio using synthesizers")


if __name__ == "__main__":
    main()
