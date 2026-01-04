# 🤖 FAQ Chatbot

A sophisticated Natural Language Processing (NLP) based chatbot that intelligently answers Frequently Asked Questions. This project uses TF-IDF vectorization and cosine similarity to match user queries with the most relevant FAQ answers from a customizable database.

## 📋 Features

- **NLP-Powered Matching**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity for intelligent question matching
- **Text Preprocessing**: Includes tokenization, lemmatization, and stop word removal for robust text processing
- **Customizable FAQs**: Easy to add, modify, or remove FAQ entries via JSON file format
- **Greeting & Farewell Detection**: Natural conversation flow with automatic greeting and goodbye responses
- **Help System**: Users can request help to see available topics and categories
- **Confidence Scoring**: Shows matched questions with confidence metrics to ensure accuracy
- **Category Organization**: FAQs organized by categories for intuitive knowledge management
- **Interactive CLI**: User-friendly command-line interface for easy interaction

## 📦 Tech Stack

- **Python 3.8+**
- **NLTK**: Natural Language Toolkit for text processing
- **scikit-learn**: Machine learning library for TF-IDF and similarity computations
- **NumPy**: Numerical computing library

## 🗂️ Project Structure

```
codealpha_faq chatbot/
├── data/
│   └── faqs.json              # FAQ questions and answers database (JSON format)
├── src/
│   ├── __init__.py            # Package initialization
│   ├── chatbot.py             # Core chatbot logic with NLP algorithms
│   ├── app.py                 # Flask web application
│   └── main.py                # CLI application entry point
├── static/
│   ├── style.css              # Web UI styling
│   └── script.js              # Web UI JavaScript
├── templates/
│   └── index.html             # Web UI HTML template
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the project directory**:
   ```bash
   cd "codealpha_faq chatbot"
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

### Running the Chatbot

```bash
python src/main.py
```

## 💬 Usage Examples

```
📝 You: What are your business hours?
🤖 Bot: We are open Monday to Friday from 9 AM to 6 PM...

📝 You: How do I return an item?
🤖 Bot: We offer a 30-day return policy for all unused items...

📝 You: help
🤖 Bot: I can help you with questions about: account, general, orders...

📝 You: quit
🤖 Bot: Goodbye! 👋 Thank you for chatting with me...
```

## 🔧 Customizing FAQs

Edit the `data/faqs.json` file to add or modify FAQs:

```json
{
    "faqs": [
        {
            "question": "Your question here?",
            "answer": "Your answer here.",
            "category": "category_name"
        }
    ]
}
```

### FAQ Structure

| Field | Description |
|-------|-------------|
| `question` | The FAQ question (used for matching) |
| `answer` | The response to display |
| `category` | Category for organization (e.g., "shipping", "returns") |

## 🧠 How It Works

1. **Preprocessing**: User input is preprocessed (lowercased, tokenized, lemmatized)
2. **TF-IDF Vectorization**: Converts text to numerical vectors based on term frequency
3. **Cosine Similarity**: Calculates similarity between user query and all FAQ questions
4. **Threshold Matching**: Returns the best match if similarity exceeds the threshold (default: 0.3)
5. **Response Generation**: Returns the matched answer or a fallback message

## 📊 Configuration

You can adjust the chatbot behavior by modifying parameters in `chatbot.py`:

```python
chatbot = FAQChatbot(
    faq_file_path='path/to/faqs.json',
    similarity_threshold=0.3  # Adjust matching sensitivity (0-1)
)
```

- **Higher threshold**: More strict matching, fewer false positives
- **Lower threshold**: More lenient matching, may return less relevant answers

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **NLTK**: Natural Language Toolkit for text processing
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NumPy**: Numerical computations

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements!

---

Made with ❤️ for CodeAlpha Internship
