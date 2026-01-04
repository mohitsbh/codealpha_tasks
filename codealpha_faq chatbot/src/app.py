"""
FAQ Chatbot - Flask Web Application
Web interface for the FAQ Chatbot.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from chatbot import create_chatbot

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize chatbot
chatbot = None


def get_chatbot():
    """Get or create the chatbot instance."""
    global chatbot
    if chatbot is None:
        chatbot = create_chatbot()
    return chatbot


@app.route('/')
def home():
    """Render the chat interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})
    
    bot = get_chatbot()
    response = bot.get_response(user_message)
    
    return jsonify({'response': response})


@app.route('/categories', methods=['GET'])
def get_categories():
    """Get all FAQ categories."""
    bot = get_chatbot()
    categories = bot.get_categories()
    return jsonify({'categories': sorted(categories)})


@app.route('/faqs/<category>', methods=['GET'])
def get_faqs_by_category(category):
    """Get FAQs for a specific category."""
    bot = get_chatbot()
    faqs = bot.get_faqs_by_category(category)
    return jsonify({'faqs': faqs})


if __name__ == '__main__':
    print("\n🚀 Starting FAQ Chatbot Web Server...")
    print("📍 Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
