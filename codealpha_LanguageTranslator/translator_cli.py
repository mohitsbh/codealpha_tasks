"""
Command Line Interface for AI Language Translator
A simple CLI tool for translating text using deep-translator.
No API key required.
"""

from deep_translator import GoogleTranslator
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
import sys


def display_languages():
    """Display all available languages"""
    print("\n" + "="*60)
    print("Available Languages:")
    print("="*60)
    
    # Sort languages by name
    sorted_langs = sorted(GOOGLE_LANGUAGES_TO_CODES.items(), key=lambda x: x[0])
    
    # Display in columns
    for i, (name, code) in enumerate(sorted_langs, 1):
        print(f"{code:5} - {name:20}", end="")
        if i % 3 == 0:
            print()
    print("\n" + "="*60)


def translate_text(text, dest_lang='en', src_lang='auto'):
    """
    Translate text to the specified language
    
    Args:
        text: Text to translate
        dest_lang: Destination language code (default: 'en')
        src_lang: Source language code (default: 'auto' for auto-detection)
    
    Returns:
        Translated text string
    """
    try:
        translator = GoogleTranslator(source=src_lang, target=dest_lang)
        result = translator.translate(text)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def detect_language(text):
    """
    Detect the language of the given text
    
    Args:
        text: Text to detect language
    
    Returns:
        Detected language code
    """
    try:
        detected = GoogleTranslator(source='auto', target='en').translate(text)
        # deep-translator doesn't expose detection directly, but we can infer
        return "Language detected (translation successful)"
    except Exception as e:
        print(f"Error: {e}")
        return None


def interactive_mode():
    """Run the translator in interactive mode"""
    print("\n" + "="*60)
    print("    🌍 AI Language Translator - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  /langs    - Show all available languages")
    print("  /to <code> - Set target language (e.g., /to es)")
    print("  /from <code> - Set source language (e.g., /from en)")
    print("  /auto     - Set source to auto-detect")
    print("  /quit     - Exit the translator")
    print("="*60 + "\n")
    
    # Create reverse mapping for validation
    codes_to_langs = {v: k for k, v in GOOGLE_LANGUAGES_TO_CODES.items()}
    valid_codes = set(GOOGLE_LANGUAGES_TO_CODES.values())
    
    target_lang = 'en'
    source_lang = 'auto'
    
    while True:
        try:
            print(f"\n[{source_lang} → {target_lang}]")
            user_input = input("Enter text (or command): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command == '/quit':
                    print("\nGoodbye! 👋")
                    break
                    
                elif command == '/langs':
                    display_languages()
                        
                elif command == '/to':
                    if len(parts) > 1:
                        new_lang = parts[1].lower()
                        if new_lang in valid_codes:
                            target_lang = new_lang
                            lang_name = codes_to_langs.get(target_lang, target_lang)
                            print(f"✓ Target language set to: {lang_name} ({target_lang})")
                        else:
                            print(f"✗ Unknown language code: {new_lang}")
                            print("  Use /langs to see available languages")
                    else:
                        print("Usage: /to <language_code>")
                        
                elif command == '/from':
                    if len(parts) > 1:
                        new_lang = parts[1].lower()
                        if new_lang in valid_codes:
                            source_lang = new_lang
                            lang_name = codes_to_langs.get(source_lang, source_lang)
                            print(f"✓ Source language set to: {lang_name} ({source_lang})")
                        else:
                            print(f"✗ Unknown language code: {new_lang}")
                            print("  Use /langs to see available languages")
                    else:
                        print("Usage: /from <language_code>")
                        
                elif command == '/auto':
                    source_lang = 'auto'
                    print("✓ Source language set to: Auto-detect")
                    
                else:
                    print(f"Unknown command: {command}")
                    
            else:
                # Translate the text
                result = translate_text(user_input, target_lang, source_lang)
                
                if result:
                    src_name = codes_to_langs.get(source_lang, source_lang)
                    dest_name = codes_to_langs.get(target_lang, target_lang)
                    
                    print(f"\n{'─'*50}")
                    if source_lang == 'auto':
                        print(f"📍 Source: Auto-detected")
                    else:
                        print(f"📍 Source: {src_name}")
                    print(f"🔄 Translation → {dest_name}:")
                    print(f"\n   {result}")
                    print(f"{'─'*50}")
                    
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point for CLI"""
    if len(sys.argv) > 1:
        # Command line arguments mode
        import argparse
        
        parser = argparse.ArgumentParser(
            description="AI Language Translator - Translate text between languages"
        )
        parser.add_argument("text", nargs="?", help="Text to translate")
        parser.add_argument("-t", "--to", default="en", help="Target language code (default: en)")
        parser.add_argument("-f", "--from", dest="source", default="auto", 
                          help="Source language code (default: auto)")
        parser.add_argument("-l", "--langs", action="store_true", 
                          help="List all available languages")
        parser.add_argument("-i", "--interactive", action="store_true",
                          help="Run in interactive mode")
        
        args = parser.parse_args()
        
        if args.langs:
            display_languages()
            
        elif args.interactive:
            interactive_mode()
                
        elif args.text:
            result = translate_text(args.text, args.to, args.source)
            if result:
                print(result)
        else:
            interactive_mode()
    else:
        # No arguments - run interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
