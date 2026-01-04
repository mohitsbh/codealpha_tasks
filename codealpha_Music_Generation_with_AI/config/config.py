"""
Configuration settings for Music Generation with AI
"""

# Model Configuration
MODEL_CONFIG = {
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "sequence_length": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 100,
    "temperature": 1.0,  # Higher = more random, Lower = more conservative
}

# Music Configuration
MUSIC_CONFIG = {
    "notes_range": 128,  # MIDI notes range (0-127)
    "velocity_default": 100,
    "tempo": 120,  # BPM
    "time_signature": (4, 4),
    "ticks_per_beat": 480,
}

# Data Configuration
DATA_CONFIG = {
    "midi_folder": "data/midi_files",
    "processed_data_folder": "data/processed",
    "model_save_path": "models/music_generator.pth",
    "generated_output_folder": "output",
}

# Generation Configuration
GENERATION_CONFIG = {
    "default_length": 500,  # Number of notes to generate
    "seed_length": 50,  # Initial seed sequence length
    "output_format": "midi",  # midi or wav
}
