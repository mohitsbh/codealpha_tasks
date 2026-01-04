"""
Data processing utilities for MIDI files
Handles loading, parsing, and preparing music data for training
"""

import os
import glob
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter

try:
    from music21 import converter, instrument, note, chord, stream
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 not installed. Install with: pip install music21")

try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False
    print("Warning: midiutil not installed. Install with: pip install midiutil")


class MidiProcessor:
    """Process MIDI files for music generation"""
    
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.note_to_int: Dict[str, int] = {}
        self.int_to_note: Dict[int, str] = {}
        self.n_vocab: int = 0
        
    def parse_midi_file(self, filepath: str) -> List[str]:
        """
        Parse a single MIDI file and extract notes/chords
        
        Args:
            filepath: Path to the MIDI file
            
        Returns:
            List of note/chord strings
        """
        if not MUSIC21_AVAILABLE:
            raise ImportError("music21 is required for MIDI parsing")
            
        notes = []
        
        try:
            midi = converter.parse(filepath)
            
            # Try multiple approaches to extract notes
            notes_to_parse = None
            
            # First try: partition by instrument
            try:
                parts = instrument.partitionByInstrument(midi)
                if parts and len(parts.parts) > 0:
                    # Collect notes from all parts
                    for part in parts.parts:
                        for element in part.recurse():
                            if isinstance(element, note.Note):
                                notes.append(str(element.pitch))
                            elif isinstance(element, chord.Chord):
                                notes.append('.'.join(str(n) for n in element.normalOrder))
            except:
                pass
            
            # Second try: flat notes if no notes found yet
            if not notes:
                for element in midi.flat.notes:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            
            # Third try: recurse through all elements
            if not notes:
                for element in midi.recurse():
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                    
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
        return notes
    
    def parse_midi_folder(self, folder_path: str) -> List[str]:
        """
        Parse all MIDI files in a folder
        
        Args:
            folder_path: Path to folder containing MIDI files
            
        Returns:
            Combined list of all notes/chords
        """
        all_notes = []
        midi_files = glob.glob(os.path.join(folder_path, "*.mid")) + \
                     glob.glob(os.path.join(folder_path, "*.midi"))
        
        print(f"Found {len(midi_files)} MIDI files")
        
        for i, file in enumerate(midi_files):
            print(f"Parsing file {i+1}/{len(midi_files)}: {os.path.basename(file)}")
            notes = self.parse_midi_file(file)
            all_notes.extend(notes)
            
        print(f"Total notes extracted: {len(all_notes)}")
        return all_notes
    
    def create_vocabulary(self, notes: List[str]) -> None:
        """
        Create note-to-integer and integer-to-note mappings
        
        Args:
            notes: List of note/chord strings
        """
        unique_notes = sorted(set(notes))
        self.n_vocab = len(unique_notes)
        
        self.note_to_int = {note: i for i, note in enumerate(unique_notes)}
        self.int_to_note = {i: note for i, note in enumerate(unique_notes)}
        
        print(f"Vocabulary size: {self.n_vocab}")
    
    def prepare_sequences(self, notes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences and corresponding outputs for training
        
        Args:
            notes: List of note/chord strings
            
        Returns:
            Tuple of (input_sequences, output_labels)
        """
        if not self.note_to_int:
            self.create_vocabulary(notes)
            
        network_input = []
        network_output = []
        
        # Create input sequences and corresponding outputs
        for i in range(0, len(notes) - self.sequence_length):
            sequence_in = notes[i:i + self.sequence_length]
            sequence_out = notes[i + self.sequence_length]
            
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
            
        n_patterns = len(network_input)
        print(f"Total training patterns: {n_patterns}")
        
        # Reshape and normalize input
        network_input = np.array(network_input)
        network_output = np.array(network_output)
        
        return network_input, network_output
    
    def save_processed_data(self, filepath: str, notes: List[str], 
                           network_input: np.ndarray, network_output: np.ndarray) -> None:
        """Save processed data to file"""
        data = {
            'notes': notes,
            'network_input': network_input,
            'network_output': network_output,
            'note_to_int': self.note_to_int,
            'int_to_note': self.int_to_note,
            'n_vocab': self.n_vocab,
            'sequence_length': self.sequence_length
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filepath: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Load processed data from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.note_to_int = data['note_to_int']
        self.int_to_note = data['int_to_note']
        self.n_vocab = data['n_vocab']
        self.sequence_length = data['sequence_length']
        
        return data['notes'], data['network_input'], data['network_output']


class MidiGenerator:
    """Generate MIDI files from note sequences"""
    
    def __init__(self, tempo: int = 120, time_signature: Tuple[int, int] = (4, 4)):
        self.tempo = tempo
        self.time_signature = time_signature
        
    def create_midi_with_music21(self, notes: List[str], output_path: str) -> None:
        """
        Create MIDI file using music21
        
        Args:
            notes: List of note/chord strings
            output_path: Path to save the MIDI file
        """
        if not MUSIC21_AVAILABLE:
            raise ImportError("music21 is required for MIDI generation")
            
        output_notes = []
        offset = 0
        
        for pattern in notes:
            # Pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    try:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        chord_notes.append(new_note)
                    except:
                        pass
                if chord_notes:
                    new_chord = chord.Chord(chord_notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
            # Pattern is a note
            else:
                try:
                    new_note = note.Note(pattern)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                except:
                    pass
                    
            offset += 0.5  # Increase offset for next note
            
        midi_stream = stream.Stream(output_notes)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        midi_stream.write('midi', fp=output_path)
        print(f"Generated MIDI saved to: {output_path}")
        
    def create_midi_with_midiutil(self, notes: List[int], output_path: str,
                                   duration: float = 0.5, velocity: int = 100) -> None:
        """
        Create MIDI file using midiutil (for numeric note values)
        
        Args:
            notes: List of MIDI note numbers (0-127)
            output_path: Path to save the MIDI file
            duration: Duration of each note in beats
            velocity: Note velocity (volume)
        """
        if not MIDIUTIL_AVAILABLE:
            raise ImportError("midiutil is required for MIDI generation")
            
        midi = MIDIFile(1)  # One track
        track = 0
        channel = 0
        time = 0
        
        midi.addTempo(track, 0, self.tempo)
        
        for pitch in notes:
            if isinstance(pitch, int) and 0 <= pitch <= 127:
                midi.addNote(track, channel, pitch, time, duration, velocity)
                time += duration
                
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            midi.writeFile(f)
        print(f"Generated MIDI saved to: {output_path}")


def create_sample_data(output_folder: str, num_samples: int = 5) -> None:
    """
    Create sample MIDI files for testing when no real data is available
    Uses music21 for better compatibility
    
    Args:
        output_folder: Folder to save sample MIDI files
        num_samples: Number of sample files to create
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Common musical patterns (MIDI note numbers)
    patterns = [
        [60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
        [60, 64, 67, 72, 67, 64, 60],  # C major arpeggio
        [69, 71, 72, 74, 76, 77, 79, 81],  # A minor scale
        [60, 63, 67, 63, 60, 63, 67, 63],  # Simple pattern
        [72, 71, 69, 67, 65, 64, 62, 60],  # Descending scale
        [60, 67, 64, 72, 60, 67, 64, 72],  # Broken chord
        [65, 64, 62, 60, 62, 64, 65, 67],  # Melodic pattern
        [60, 60, 67, 67, 69, 69, 67],  # Twinkle pattern
    ]
    
    if MUSIC21_AVAILABLE:
        # Use music21 for better MIDI compatibility
        for i in range(num_samples):
            s = stream.Stream()
            
            # Create a longer sequence by combining patterns
            for _ in range(16):  # More repetitions for more training data
                pattern = patterns[np.random.randint(0, len(patterns))]
                transposition = np.random.choice([-12, -7, -5, 0, 5, 7, 12])
                
                for pitch_num in pattern:
                    adjusted_pitch = pitch_num + transposition
                    # Keep in valid MIDI range
                    adjusted_pitch = max(21, min(108, adjusted_pitch))
                    n = note.Note(adjusted_pitch)
                    n.quarterLength = np.random.choice([0.25, 0.5, 0.5, 1.0])
                    s.append(n)
                    
                # Occasionally add chords
                if np.random.random() > 0.7:
                    chord_pitches = [60, 64, 67]  # C major chord
                    chord_notes = []
                    for p in chord_pitches:
                        adjusted = max(21, min(108, p + transposition))
                        chord_notes.append(note.Note(adjusted))
                    c = chord.Chord(chord_notes)
                    c.quarterLength = 1.0
                    s.append(c)
            
            output_path = os.path.join(output_folder, f"sample_{i+1}.mid")
            s.write('midi', fp=output_path)
            print(f"Generated MIDI saved to: {output_path}")
            
    elif MIDIUTIL_AVAILABLE:
        # Fallback to midiutil
        generator = MidiGenerator(tempo=120)
        
        for i in range(num_samples):
            sequence = []
            for _ in range(16):
                pattern = patterns[np.random.randint(0, len(patterns))]
                transposition = np.random.choice([-12, -7, -5, 0, 5, 7, 12])
                sequence.extend([max(21, min(108, n + transposition)) for n in pattern])
                
            output_path = os.path.join(output_folder, f"sample_{i+1}.mid")
            generator.create_midi_with_midiutil(sequence, output_path)
    else:
        print("Neither music21 nor midiutil available. Cannot create sample data.")
        return
        
    print(f"Created {num_samples} sample MIDI files in {output_folder}")


if __name__ == "__main__":
    # Test the data processing
    print("Testing data processing utilities...")
    
    # Create sample data
    create_sample_data("data/midi_files", num_samples=3)
    
    # Test processor
    processor = MidiProcessor(sequence_length=50)
    
    if MUSIC21_AVAILABLE:
        notes = processor.parse_midi_folder("data/midi_files")
        if notes:
            network_input, network_output = processor.prepare_sequences(notes)
            print(f"Input shape: {network_input.shape}")
            print(f"Output shape: {network_output.shape}")
    else:
        print("Install music21 to test MIDI parsing: pip install music21")
