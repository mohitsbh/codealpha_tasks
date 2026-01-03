"""
FAQ Chatbot - Main Application
Command-line interface for the FAQ Chatbot.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import create_chatbot, FAQChatbot


def print_banner():
    """Print the chatbot welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║               🤖 FAQ CHATBOT ASSISTANT 🤖                   ║
║                                                              ║
║     Ask me anything! Type 'help' for available topics.       ║
║     Type 'quit' or 'exit' to end the conversation.           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_cli_chatbot(chatbot: FAQChatbot):
    """
    Run the chatbot in CLI mode.
    
    Args:
        chatbot: Initialized FAQChatbot instance
    """
    print_banner()
    print("\n" + chatbot.get_response("hello") + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()
            
            # Check for empty input
            if not user_input:
                print("\n🤖 Bot: Please type a question or 'help' for assistance.")
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: " + chatbot.get_response("goodbye"))
                break
            
            # Get and display response
            response = chatbot.get_response(user_input)
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: " + chatbot.get_response("goodbye"))
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")


def main():
    """Main entry point for the FAQ Chatbot."""
    print("\n🔄 Initializing FAQ Chatbot...\n")
    
    try:
        # Create the chatbot instance
        chatbot = create_chatbot()
        
        # Run the CLI interface
        run_cli_chatbot(chatbot)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the FAQ data file exists in the 'data' folder.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
