"""
FAQ Chatbot - Core Module
This module contains the main chatbot logic using NLP techniques
for matching user queries to FAQ questions.
"""

import json
import os
import re
import string
from typing import Optional, Tuple, List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class FAQChatbot:
    """
    A chatbot that answers frequently asked questions using NLP techniques.
    Uses TF-IDF vectorization and cosine similarity for question matching.
    """
    
    def __init__(self, faq_file_path: str, similarity_threshold: float = 0.3):
        """
        Initialize the FAQ Chatbot.
        
        Args:
            faq_file_path: Path to the JSON file containing FAQs
            similarity_threshold: Minimum similarity score to consider a match (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set()
        self.faqs: List[Dict] = []
        self.questions: List[str] = []
        self.processed_questions: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load stop words after download
        self.stop_words = set(stopwords.words('english'))
        
        # Load and process FAQs
        self._load_faqs(faq_file_path)
        self._build_tfidf_matrix()
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages."""
        nltk_packages = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        for package in nltk_packages:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
    
    def _load_faqs(self, file_path: str) -> None:
        """
        Load FAQ data from JSON file.
        
        Args:
            file_path: Path to the FAQ JSON file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAQ file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.faqs = data.get('faqs', [])
        self.questions = [faq['question'] for faq in self.faqs]
        self.processed_questions = [self._preprocess_text(q) for q in self.questions]
        
        print(f"✓ Loaded {len(self.faqs)} FAQ entries")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP analysis.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalpha()
        ]
        
        return ' '.join(processed_tokens)
    
    def _build_tfidf_matrix(self) -> None:
        """Build TF-IDF matrix from processed questions."""
        if self.processed_questions:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
            print("✓ TF-IDF matrix built successfully")
    
    def find_best_match(self, user_query: str) -> Tuple[Optional[Dict], float]:
        """
        Find the best matching FAQ for a user query.
        
        Args:
            user_query: The user's question
            
        Returns:
            Tuple of (matching FAQ dict or None, similarity score)
        """
        # Preprocess the user query
        processed_query = self._preprocess_text(user_query)
        
        if not processed_query.strip():
            return None, 0.0
        
        # Transform the query using the fitted vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Find the best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.similarity_threshold:
            return self.faqs[best_idx], best_score
        
        return None, best_score
    
    def get_response(self, user_query: str) -> str:
        """
        Get a response for the user's query.
        
        Args:
            user_query: The user's question
            
        Returns:
            The chatbot's response
        """
        # Check for greeting
        if self._is_greeting(user_query):
            return self._get_greeting_response()
        
        # Check for farewell
        if self._is_farewell(user_query):
            return self._get_farewell_response()
        
        # Check for help request
        if self._is_help_request(user_query):
            return self._get_help_response()
        
        # Find the best matching FAQ
        match, score = self.find_best_match(user_query)
        
        if match:
            response = f"{match['answer']}"
            if score < 0.5:
                response = f"I think you're asking about: \"{match['question']}\"\n\n{response}"
            return response
        
        return self._get_fallback_response()
    
    def _is_greeting(self, text: str) -> bool:
        """Check if the text is a greeting."""
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 
                     'good afternoon', 'good evening', 'howdy', 'hiya']
        text_lower = text.lower().strip()
        return any(g in text_lower for g in greetings)
    
    def _is_farewell(self, text: str) -> bool:
        """Check if the text is a farewell."""
        farewells = ['bye', 'goodbye', 'see you', 'farewell', 'quit', 
                     'exit', 'thanks bye', 'thank you bye']
        text_lower = text.lower().strip()
        return any(f in text_lower for f in farewells)
    
    def _is_help_request(self, text: str) -> bool:
        """Check if the user is asking for help."""
        help_phrases = ['help', 'what can you do', 'how do you work', 
                        'what do you know', 'list questions', 'show topics']
        text_lower = text.lower().strip()
        return any(h in text_lower for h in help_phrases)
    
    def _get_greeting_response(self) -> str:
        """Get a greeting response."""
        return ("Hello! 👋 I'm your FAQ Assistant. I can help answer your questions "
                "about our services, orders, shipping, returns, and more.\n\n"
                "Feel free to ask me anything, or type 'help' to see what topics I can assist with!")
    
    def _get_farewell_response(self) -> str:
        """Get a farewell response."""
        return ("Goodbye! 👋 Thank you for chatting with me. "
                "If you have more questions later, feel free to come back anytime. "
                "Have a great day!")
    
    def _get_help_response(self) -> str:
        """Get a help response listing available topics."""
        categories = set(faq.get('category', 'general') for faq in self.faqs)
        categories_list = ', '.join(sorted(categories))
        
        sample_questions = [
            "• What are your business hours?",
            "• How do I track my order?",
            "• What is your return policy?",
            "• Do you offer international shipping?",
            "• How do I reset my password?"
        ]
        
        return (f"I can help you with questions about: {categories_list}\n\n"
                f"Here are some example questions you can ask:\n"
                f"{chr(10).join(sample_questions)}\n\n"
                f"Just type your question and I'll do my best to help!")
    
    def _get_fallback_response(self) -> str:
        """Get a fallback response when no match is found."""
        return ("I'm sorry, I couldn't find a good answer to your question. 🤔\n\n"
                "Could you try:\n"
                "• Rephrasing your question\n"
                "• Using different keywords\n"
                "• Typing 'help' to see available topics\n\n"
                "Or contact our customer support at support@example.com for personalized assistance.")
    
    def get_categories(self) -> List[str]:
        """Get list of all FAQ categories."""
        return list(set(faq.get('category', 'general') for faq in self.faqs))
    
    def get_faqs_by_category(self, category: str) -> List[Dict]:
        """Get all FAQs in a specific category."""
        return [faq for faq in self.faqs if faq.get('category') == category]


def create_chatbot(faq_file: str = None) -> FAQChatbot:
    """
    Factory function to create a chatbot instance.
    
    Args:
        faq_file: Path to FAQ JSON file. If None, uses default path.
        
    Returns:
        Configured FAQChatbot instance
    """
    if faq_file is None:
        # Default to the data folder relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faq_file = os.path.join(base_dir, 'data', 'faqs.json')
    
    return FAQChatbot(faq_file)
