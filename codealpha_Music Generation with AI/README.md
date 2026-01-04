# Music Generation with AI 🎵

An AI-powered music generation system using deep learning models (LSTM, Transformer, VAE) to create original musical compositions.

## 🎯 Features

- **Multiple Model Architectures:**
  - **LSTM**: Long Short-Term Memory networks for sequential music generation
  - **Transformer**: Self-attention based model for capturing long-range dependencies
  - **VAE**: Variational Autoencoder for learning latent music representations

- **MIDI Processing**: Full support for parsing and generating MIDI files
- **Temperature Sampling**: Control creativity vs. coherence in generation
- **Top-k/Top-p Sampling**: Advanced sampling strategies for diverse outputs
- **Easy Training**: Simple command-line interface for model training
- **Demo Mode**: Quick demonstration without trained models

## 📁 Project Structure

```
Music Generation with AI/
├── config/
│   ├── __init__.py
│   └── config.py           # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # MIDI parsing and data preparation
│   ├── models.py           # Neural network architectures
│   ├── train.py            # Training script
│   └── generate.py         # Music generation script
├── data/
│   ├── midi_files/         # Input MIDI files for training
│   └── processed/          # Processed data cache
├── models/                  # Saved model checkpoints
├── output/                  # Generated music files
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd "Music Generation with AI"

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
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
