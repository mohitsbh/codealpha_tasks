"""
Main entry point for Music Generation with AI
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Music Generation with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train with default settings
  python main.py train --epochs 50        # Train for 50 epochs
  python main.py generate                 # Generate music
  python main.py generate --demo          # Generate demo without model
  python main.py generate --temperature 0.8 --length 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a music generation model')
    train_parser.add_argument('--model', type=str, default='lstm',
                             choices=['lstm', 'transformer', 'vae'])
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--create_sample_data', action='store_true')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate music')
    gen_parser.add_argument('--temperature', type=float, default=1.0)
    gen_parser.add_argument('--length', type=int, default=500)
    gen_parser.add_argument('--output', type=str, default=None)
    gen_parser.add_argument('--demo', action='store_true')
    
    # Demo command
    subparsers.add_parser('demo', help='Create demo music files')
    
    # UI command
    subparsers.add_parser('ui', help='Launch web UI')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.train import main as train_main
        sys.argv = ['train.py']
        if args.model:
            sys.argv.extend(['--model', args.model])
        if args.epochs:
            sys.argv.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(['--batch_size', str(args.batch_size)])
        if args.create_sample_data:
            sys.argv.append('--create_sample_data')
        train_main()
        
    elif args.command == 'generate':
        from src.generate import main as gen_main
        sys.argv = ['generate.py']
        if args.temperature:
            sys.argv.extend(['--temperature', str(args.temperature)])
        if args.length:
            sys.argv.extend(['--length', str(args.length)])
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.demo:
            sys.argv.append('--demo')
        gen_main()
        
    elif args.command == 'demo':
        from src.generate import create_demo_generation
        create_demo_generation('output')
    
    elif args.command == 'ui':
        from src.ui import main as ui_main
        ui_main()
        
    else:
        parser.print_help()
        print("\n✨ Quick Start:")
        print("  1. python main.py demo           # Create demo music")
        print("  2. python main.py train          # Train a model")
        print("  3. python main.py generate       # Generate music")
        print("  4. python main.py ui             # Launch web UI")


if __name__ == "__main__":
    main()
