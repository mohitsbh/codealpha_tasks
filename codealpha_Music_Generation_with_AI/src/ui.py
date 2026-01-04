"""
Web UI for Music Generation with AI
Uses Gradio for an interactive interface
"""

import os
import sys
from datetime import datetime
from typing import Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

from config import DATA_CONFIG, MUSIC_CONFIG


def check_model_exists() -> bool:
    """Check if a trained model exists"""
    return os.path.exists(DATA_CONFIG['model_save_path']) and \
           os.path.exists('models/vocabulary.pkl')


def generate_with_ai_model(
    length: int,
    temperature: float,
    top_k: int
) -> Tuple[Optional[str], str]:
    """Generate music using the trained AI model"""
    
    if not check_model_exists():
        return None, "❌ No trained model found! Please train a model first or use Demo mode."
    
    try:
        from src.generate import MusicGenerator
        
        generator = MusicGenerator(
            model_path=DATA_CONFIG['model_save_path'],
            vocab_path='models/vocabulary.pkl',
            model_type='lstm'
        )
        
        # Generate output path
        os.makedirs(DATA_CONFIG['generated_output_folder'], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            DATA_CONFIG['generated_output_folder'],
            f"ai_generated_{timestamp}.mid"
        )
        
        # Generate music
        output_path = generator.generate_and_save(
            output_path=output_path,
            length=length,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None
        )
        
        return output_path, f"✅ Successfully generated {length} notes!\n\n📁 Saved to: {output_path}\n\n🎹 Open the MIDI file with any music player or DAW."
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_demo_music(
    pattern_type: str,
    num_repeats: int,
    tempo: int,
    add_variation: bool
) -> Tuple[Optional[str], str]:
    """Generate demo music without trained model"""
    
    try:
        from src.data_processing import MidiGenerator
        
        # Musical patterns (MIDI note numbers)
        patterns = {
            'C Major Scale': [60, 62, 64, 65, 67, 69, 71, 72],
            'C Minor Scale': [60, 62, 63, 65, 67, 68, 70, 72],
            'G Major Scale': [67, 69, 71, 72, 74, 76, 78, 79],
            'A Minor Scale': [69, 71, 72, 74, 76, 77, 79, 81],
            'C Major Arpeggio': [60, 64, 67, 72, 67, 64],
            'Pentatonic': [60, 62, 64, 67, 69, 72, 69, 67, 64, 62],
            'Blues Scale': [60, 63, 65, 66, 67, 70, 72, 70, 67, 66, 65, 63],
            'Twinkle Twinkle': [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60],
            'Ode to Joy': [64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62],
        }
        
        generator = MidiGenerator(tempo=tempo)
        
        os.makedirs(DATA_CONFIG['generated_output_folder'], exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = pattern_type.lower().replace(' ', '_')
        output_path = os.path.join(
            DATA_CONFIG['generated_output_folder'],
            f"demo_{safe_name}_{timestamp}.mid"
        )
        
        pattern = patterns[pattern_type]
        sequence = []
        
        for rep in range(num_repeats):
            current_pattern = pattern.copy()
            
            if add_variation and rep > 0:
                # Add octave transposition
                transposition = np.random.choice([-12, 0, 0, 12])
                current_pattern = [max(36, min(96, n + transposition)) for n in current_pattern]
            
            # Add ascending pattern
            sequence.extend(current_pattern)
            
            # Add descending pattern (excluding first note to avoid repetition)
            if rep < num_repeats - 1:
                sequence.extend(current_pattern[::-1][1:-1])
        
        generator.create_midi_with_midiutil(sequence, output_path)
        
        return output_path, f"""✅ Demo music created!

🎵 Pattern: {pattern_type}
🎼 Notes: {len(sequence)}
🎹 Tempo: {tempo} BPM
🔄 Repeats: {num_repeats}

📁 Saved to: {output_path}

💡 Open the MIDI file with any music player or DAW (FL Studio, Ableton, GarageBand, etc.)"""
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_random_melody(
    num_notes: int,
    scale: str,
    tempo: int
) -> Tuple[Optional[str], str]:
    """Generate a random melody"""
    
    try:
        from src.data_processing import MidiGenerator
        
        scales = {
            'C Major': [60, 62, 64, 65, 67, 69, 71, 72, 74, 76],
            'C Minor': [60, 62, 63, 65, 67, 68, 70, 72, 74, 75],
            'C Pentatonic': [60, 62, 64, 67, 69, 72, 74, 76, 79],
            'C Blues': [60, 63, 65, 66, 67, 70, 72, 75, 77],
            'Chromatic': list(range(60, 73)),
        }
        
        scale_notes = scales[scale]
        generator = MidiGenerator(tempo=tempo)
        
        os.makedirs(DATA_CONFIG['generated_output_folder'], exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            DATA_CONFIG['generated_output_folder'],
            f"random_melody_{timestamp}.mid"
        )
        
        # Generate random melody with some musical logic
        sequence = []
        current_note = np.random.choice(scale_notes[:4])  # Start in lower range
        
        for i in range(num_notes):
            sequence.append(current_note)
            
            # Movement probability - prefer small intervals
            movement = np.random.choice([-2, -1, -1, 0, 1, 1, 2])
            
            # Find next note in scale
            try:
                current_idx = scale_notes.index(current_note)
                next_idx = max(0, min(len(scale_notes) - 1, current_idx + movement))
                current_note = scale_notes[next_idx]
            except ValueError:
                current_note = np.random.choice(scale_notes)
        
        generator.create_midi_with_midiutil(sequence, output_path)
        
        return output_path, f"""✅ Random melody generated!

🎲 Scale: {scale}
🎼 Notes: {num_notes}
🎹 Tempo: {tempo} BPM

📁 Saved to: {output_path}"""
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_ui():
    """Create the Gradio UI"""
    
    with gr.Blocks(
        title="🎵 Music Generation with AI",
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue")
    ) as app:
        
        gr.Markdown(
            """
            # 🎵 Music Generation with AI
            
            Create music using AI or explore musical patterns! Generate MIDI files that you can play or import into any DAW.
            """
        )
        
        with gr.Tabs():
            
            # === TAB 1: AI Generation ===
            with gr.TabItem("🤖 AI Generation"):
                model_status = "✅ Model Ready!" if check_model_exists() else "⚠️ No trained model found"
                gr.Markdown(f"**Status:** {model_status}")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        ai_length = gr.Slider(
                            minimum=50, maximum=1000, value=300, step=50,
                            label="Number of Notes",
                            info="How many notes to generate"
                        )
                        ai_temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                            label="Temperature",
                            info="Lower=Conservative, Higher=Creative"
                        )
                        ai_top_k = gr.Slider(
                            minimum=0, maximum=50, value=20, step=5,
                            label="Top-K Sampling",
                            info="0=Disabled, Higher=More variety"
                        )
                        ai_generate_btn = gr.Button("🎵 Generate with AI", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        ai_output_file = gr.File(label="Generated MIDI File")
                        ai_status = gr.Textbox(label="Status", lines=6, interactive=False)
                
                ai_generate_btn.click(
                    fn=generate_with_ai_model,
                    inputs=[ai_length, ai_temperature, ai_top_k],
                    outputs=[ai_output_file, ai_status]
                )
            
            # === TAB 2: Demo Patterns ===
            with gr.TabItem("🎹 Demo Patterns"):
                gr.Markdown("Generate music from preset patterns - **no training required!**")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        demo_pattern = gr.Dropdown(
                            choices=[
                                'C Major Scale', 'C Minor Scale', 'G Major Scale',
                                'A Minor Scale', 'C Major Arpeggio', 'Pentatonic',
                                'Blues Scale', 'Twinkle Twinkle', 'Ode to Joy'
                            ],
                            value='C Major Scale',
                            label="Pattern"
                        )
                        demo_repeats = gr.Slider(
                            minimum=1, maximum=8, value=4, step=1,
                            label="Repeats"
                        )
                        demo_tempo = gr.Slider(
                            minimum=60, maximum=180, value=120, step=10,
                            label="Tempo (BPM)"
                        )
                        demo_variation = gr.Checkbox(
                            value=True,
                            label="Add Octave Variations"
                        )
                        demo_generate_btn = gr.Button("🎼 Generate Pattern", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        demo_output_file = gr.File(label="Generated MIDI File")
                        demo_status = gr.Textbox(label="Status", lines=8, interactive=False)
                
                demo_generate_btn.click(
                    fn=generate_demo_music,
                    inputs=[demo_pattern, demo_repeats, demo_tempo, demo_variation],
                    outputs=[demo_output_file, demo_status]
                )
            
            # === TAB 3: Random Melody ===
            with gr.TabItem("🎲 Random Melody"):
                gr.Markdown("Generate random melodies based on musical scales")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        rand_notes = gr.Slider(
                            minimum=20, maximum=200, value=64, step=8,
                            label="Number of Notes"
                        )
                        rand_scale = gr.Dropdown(
                            choices=['C Major', 'C Minor', 'C Pentatonic', 'C Blues', 'Chromatic'],
                            value='C Pentatonic',
                            label="Scale"
                        )
                        rand_tempo = gr.Slider(
                            minimum=60, maximum=180, value=100, step=10,
                            label="Tempo (BPM)"
                        )
                        rand_generate_btn = gr.Button("🎲 Generate Random", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        rand_output_file = gr.File(label="Generated MIDI File")
                        rand_status = gr.Textbox(label="Status", lines=6, interactive=False)
                
                rand_generate_btn.click(
                    fn=generate_random_melody,
                    inputs=[rand_notes, rand_scale, rand_tempo],
                    outputs=[rand_output_file, rand_status]
                )
            
            # === TAB 4: Help ===
            with gr.TabItem("❓ Help"):
                gr.Markdown(
                    """
                    ## 📖 How to Use
                    
                    ### 🤖 AI Generation
                    Uses a trained neural network (LSTM) to generate music. Requires training first.
                    
                    - **Temperature**: Controls creativity
                      - Low (0.1-0.5): More repetitive, safer choices
                      - Medium (0.6-1.0): Balanced
                      - High (1.0-2.0): More random, experimental
                    
                    - **Top-K**: Limits choices to top K most likely notes
                    
                    ### 🎹 Demo Patterns  
                    Pre-defined musical patterns - great for testing or learning!
                    
                    ### 🎲 Random Melody
                    Generates random notes within a musical scale.
                    
                    ---
                    
                    ## 🎵 Playing MIDI Files
                    
                    Generated `.mid` files can be played with:
                    - **Windows Media Player** (with MIDI support)
                    - **VLC Media Player**
                    - **Online**: [Signal MIDI Player](https://signal.vercel.app/)
                    - **DAWs**: FL Studio, Ableton, GarageBand, LMMS
                    
                    ---
                    
                    ## 🚀 Training Your Own Model
                    
                    ```bash
                    # Create sample data and train
                    python src/train.py --create_sample_data --epochs 30
                    
                    # Or with your own MIDI files
                    # Place MIDI files in data/midi_files/ then:
                    python src/train.py --epochs 50
                    ```
                    """
                )
        
        gr.Markdown(
            """
            ---
            Made with ❤️ using PyTorch and Gradio | [GitHub](https://github.com)
            """
        )
    
    return app


def main():
    """Launch the UI"""
    app = create_ui()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
