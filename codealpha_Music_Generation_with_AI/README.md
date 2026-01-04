# 🎵 Music Generation with AI

An advanced AI-powered music generation system using deep learning models (LSTM, Transformer, VAE) to create original musical compositions. This project combines neural networks with MIDI processing to generate creative, coherent musical pieces.

## 🎯 Features

### Model Architectures

- **LSTM** (Long Short-Term Memory): 
  - Sequential music generation with memory of previous notes
  - Excellent for capturing musical patterns and dependencies
  - Fast training and inference

- **Transformer**: 
  - Self-attention mechanism for capturing long-range dependencies
  - Better at understanding complex musical structures
  - State-of-the-art approach to sequence generation

- **VAE** (Variational Autoencoder): 
  - Learns latent representations of music
  - Enables interpolation between different musical styles
  - Generates diverse outputs from continuous space

### Key Capabilities

- **MIDI Processing**: Complete support for parsing, processing, and generating MIDI files
- **Temperature Sampling**: Control creativity vs. coherence in generation (0.0 = deterministic, 1.0+ = creative)
- **Top-k/Top-p Sampling**: Advanced sampling strategies for diverse and coherent outputs
- **Easy Training**: Simple command-line interface for model training with customizable parameters
- **Demo Mode**: Quick demonstration without requiring pre-trained models
- **Multiple Sampling Strategies**: Diverse options for controlling output characteristics
- **Configurable Parameters**: Extensive settings for training and generation

## 📦 Tech Stack

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **Music21**: Music processing and MIDI handling
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization (for analysis)

## 🗂️ Project Structure

```
codealpha_Music Generation with AI/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration settings and hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # MIDI parsing and data preparation
│   ├── models.py              # Neural network architectures (LSTM, Transformer, VAE)
│   ├── train.py               # Training script and training loops
│   ├── generate.py            # Music generation script and utilities
│   ├── ui.py                  # User interface utilities
│   └── __pycache__/           # Python cache files
├── data/
│   └── midi_files/            # Input MIDI files for training (10 samples included)
├── models/
│   └── music_generator.pth    # Pre-trained model checkpoint
├── output/                    # Generated music files (MIDI format)
├── config.yaml                # YAML configuration file
├── requirements.txt           # Python dependencies
├── main.py                    # Main entry point with CLI
└── README.md                  # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- At least 2GB of free disk space for models and data

### Installation

1. **Navigate to the project directory**:
   ```bash
   cd "codealpha_Music Generation with AI"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training a Model

Train a new music generation model:

```bash
# Train with default settings (LSTM model, 100 epochs)
python main.py train

# Train a Transformer model
python main.py train --model transformer --epochs 50

# Train VAE with custom batch size
python main.py train --model vae --batch_size 32 --epochs 100

# Train with learning rate customization
python main.py train --model lstm --lr 0.001 --epochs 200
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | lstm | Model architecture: `lstm`, `transformer`, or `vae` |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--seq_length` | 50 | Sequence length for training |
| `--hidden_dim` | 256 | Hidden dimension size |

### Generating Music

Generate new music compositions:

```bash
# Generate music using default settings
python main.py generate

# Generate with demo mode (no model required)
python main.py generate --demo

# Generate with custom temperature (creativity control)
python main.py generate --temperature 0.8 --length 1000

# Generate with specific model
python main.py generate --model transformer --temperature 0.7

# Generate using top-k sampling
python main.py generate --temperature 0.9 --top_k 40

# Generate using top-p (nucleus) sampling
python main.py generate --temperature 0.8 --top_p 0.95

# Save with custom filename
python main.py generate --output my_composition.mid
```

#### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | lstm | Model to use for generation |
| `--temperature` | 1.0 | Creativity level (lower = more conservative, higher = more creative) |
| `--length` | 500 | Length of generated sequence (in notes) |
| `--top_k` | None | Top-k sampling parameter (keeps top k predictions) |
| `--top_p` | None | Top-p (nucleus) sampling (keeps predictions with cumulative prob ≤ p) |
| `--output` | auto-generated | Output filename for generated MIDI |
| `--demo` | False | Run in demo mode without trained model |

### Example Commands

```bash
# Complete workflow
python main.py train --model lstm --epochs 50
python main.py generate --temperature 0.8 --length 1000

# Quick demo without training
python main.py generate --demo --temperature 1.0

# Experiment with different models
python main.py train --model transformer --epochs 100
python main.py generate --model transformer --temperature 0.9
```

## 📊 Understanding the Models

### LSTM Architecture

```
Input → Embedding → LSTM Layers → Output Layer → MIDI Notes
```

- **Pros**: Fast, good for short-term dependencies, stable training
- **Cons**: Limited long-term memory
- **Best for**: Quick experimentation and generation

### Transformer Architecture

```
Input → Embedding → Multi-Head Attention → Feed-Forward → Output
```

- **Pros**: Captures long-term dependencies, parallelizable, state-of-the-art
- **Cons**: Slower training, more memory required
- **Best for**: High-quality, coherent compositions

### VAE Architecture

```
Input → Encoder → Latent Space → Decoder → Output
```

- **Pros**: Continuous latent space, smooth interpolation possible
- **Cons**: May lose fine details, requires careful tuning
- **Best for**: Generating style variations and creative explorations

## 🎼 MIDI File Handling

### Input Files

Place MIDI files in `data/midi_files/` directory. Supported MIDI features:
- Standard MIDI note events (0-127)
- Tempo information
- Time signatures
- Velocity information

### Output Files

Generated MIDI files are saved in the `output/` directory with:
- Timestamp in filename
- Standard MIDI format (can be opened in any DAW)
- Playable with music players supporting MIDI

### Processing Pipeline

1. **MIDI Loading**: Parse MIDI files and extract note sequences
2. **Normalization**: Convert notes to a standard range (0-127)
3. **Sequencing**: Create sequences of fixed length for training
4. **Encoding**: Convert sequences to tensor format for neural networks
5. **Generation**: Generate new sequences from trained model
6. **MIDI Export**: Convert generated sequences back to MIDI format

## 🔧 Configuration

Customize behavior in `config/config.py`:

```python
# Model parameters
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100

# Generation parameters
TEMPERATURE = 1.0
MAX_SEQUENCE_LENGTH = 500
```

## 📈 Performance Tips

1. **GPU Acceleration**: Models run on GPU if available (CUDA)
2. **Data Preparation**: Larger MIDI datasets improve generation quality
3. **Epochs**: More epochs generally produce better results (diminishing returns after ~100)
4. **Batch Size**: Larger batches converge faster but need more memory
5. **Temperature**: Experiment with 0.7-1.2 for best results

## 🎹 Listening to Generated Music

Generated MIDI files in `output/` can be played with:

- **Windows**: Windows Media Player, or any MIDI-capable player
- **macOS**: GarageBand, QuickTime, or third-party MIDI players
- **Linux**: Timidity, FluidSynth, or MuseScore
- **DAWs**: Ableton Live, FL Studio, Logic Pro, etc.

## 🧪 Testing

The project includes unit tests for core components:

```bash
python -m pytest tests/ -v
```

## 📊 Output Samples

The `output/` directory contains example generated compositions:

- `ai_generated_[timestamp].mid` - Generated compositions with timestamps
- `demo_melody.mid` - Demo melody pattern
- `demo_random.mid` - Demo with random notes
- `demo_scale.mid` - Demo with musical scale

## 🚧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No MIDI files found" | Ensure MIDI files are in `data/midi_files/` |
| Out of memory error | Reduce `batch_size` or `seq_length` parameters |
| Slow generation | Use GPU-enabled PyTorch if available |
| Generated music sounds wrong | Adjust `temperature` parameter (try 0.8-1.0) |
| Model not saving | Check disk space and write permissions in `models/` |

## 🔮 Future Enhancements

- Real-time music generation in web interface
- Audio synthesis with pre-trained instruments
- Music genre classification
- Melody and harmony separation
- Interactive music composition tool
- Support for additional music formats (ABC, MuseData)
- Pre-trained models for different genres

## 📚 References

- PyTorch: https://pytorch.org/
- Music21: https://web.mit.edu/music21/
- Transformer Architecture: https://arxiv.org/abs/1706.03762
- LSTM Networks: https://en.wikipedia.org/wiki/Long_short-term_memory

## 📄 License

Open source - free for educational and personal use.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better MIDI processing
- Additional model architectures
- Performance optimizations
- Dataset collection
- Documentation improvements
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Demo Music (No Training Required)

```bash
python src/generate.py --demo
```

This creates sample MIDI files in the `output/` folder without needing a trained model.

### 3. Train a Model

Place your MIDI files in `data/midi_files/` or use sample data:

```bash
# Create sample training data and train
python src/train.py --create_sample_data --epochs 50

# Or with specific options
python src/train.py --model lstm --epochs 100 --batch_size 64
```

### 4. Generate Music

```bash
# Generate with default settings
python src/generate.py

# Custom generation
python src/generate.py --length 1000 --temperature 0.8 --output my_song.mid
```

## 🎛️ Configuration

Edit `config/config.py` to customize:

```python
MODEL_CONFIG = {
    "embedding_dim": 256,     # Note embedding dimension
    "hidden_dim": 512,        # LSTM/Transformer hidden size
    "num_layers": 3,          # Number of layers
    "dropout": 0.3,           # Dropout rate
    "sequence_length": 100,   # Input sequence length
    "batch_size": 64,         # Training batch size
    "learning_rate": 0.001,   # Learning rate
    "epochs": 100,            # Training epochs
    "temperature": 1.0,       # Sampling temperature
}
```

## 📊 Model Comparison

| Model | Strengths | Best For |
|-------|-----------|----------|
| **LSTM** | Fast training, good for short patterns | General music generation |
| **Transformer** | Captures long-range dependencies | Complex compositions |
| **VAE** | Latent space interpolation, variation | Style transfer, exploration |

## 🎹 Usage Examples

### Training with Custom Data

```python
from src.train import MusicTrainer, prepare_data
from src.models import MusicLSTM
import torch

# Prepare data
train_loader, val_loader, processor = prepare_data(
    midi_folder="data/midi_files",
    sequence_length=100,
    batch_size=64
)

# Create model
model = MusicLSTM(vocab_size=processor.n_vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train
trainer = MusicTrainer(model, device, learning_rate=0.001)
history = trainer.train(train_loader, val_loader, epochs=100)
```

### Generating Music Programmatically

```python
from src.generate import MusicGenerator

# Load trained model
generator = MusicGenerator(
    model_path="models/music_generator.pth",
    vocab_path="models/vocabulary.pkl",
    model_type="lstm"
)

# Generate
notes = generator.generate(
    length=500,
    temperature=0.8,
    top_k=50
)

# Save to MIDI
generator.generate_and_save("output/my_composition.mid", length=500)
```

## 📝 Command Line Arguments

### Training (`train.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | lstm | Model type: lstm, transformer, vae |
| `--midi_folder` | data/midi_files | Input MIDI folder |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--learning_rate` | 0.001 | Learning rate |
| `--sequence_length` | 100 | Input sequence length |
| `--create_sample_data` | False | Create sample MIDI data |

### Generation (`generate.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | models/music_generator.pth | Trained model path |
| `--model_type` | lstm | Model type |
| `--output` | auto-generated | Output MIDI file |
| `--length` | 500 | Notes to generate |
| `--temperature` | 1.0 | Sampling temperature (0.1-2.0) |
| `--top_k` | None | Top-k sampling |
| `--top_p` | None | Nucleus sampling |
| `--demo` | False | Run demo mode |

## 🎵 Temperature Guide

- **0.1 - 0.5**: Very conservative, repetitive but coherent
- **0.5 - 0.8**: Balanced creativity and structure
- **0.8 - 1.0**: More creative and varied
- **1.0 - 1.5**: High creativity, may be less coherent
- **1.5 - 2.0**: Very experimental, possibly chaotic

## 💡 Tips for Better Results

1. **Training Data**: Use high-quality MIDI files in a consistent style
2. **Data Quantity**: More training data generally improves results
3. **Sequence Length**: Longer sequences capture more context but need more data
4. **Training Duration**: Train for at least 50-100 epochs
5. **Temperature Tuning**: Start at 0.8-1.0 and adjust based on results
6. **Model Selection**: 
   - LSTM for quick experiments
   - Transformer for complex pieces
   - VAE for style exploration

## 🔧 Troubleshooting

### "music21 not found"
```bash
pip install music21
```

### "CUDA out of memory"
- Reduce batch size: `--batch_size 32`
- Reduce model size in config
- Use CPU: The code automatically falls back to CPU

### "Not enough notes for sequence length"
- Add more MIDI files
- Reduce sequence length: `--sequence_length 50`
- Use longer MIDI files

## 📚 References

- [Music21 Documentation](https://web.mit.edu/music21/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MIDI File Format](https://www.midi.org/specifications)

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new model architectures
- Improve data processing
- Add audio conversion features
- Create visualization tools

---

Made with ❤️ and AI
