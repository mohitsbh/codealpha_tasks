// FAQ Chatbot - Frontend JavaScript

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const chatForm = document.getElementById('chatForm');

/**
 * Send a message to the chatbot
 */
async function sendMessage(event) {
    event.preventDefault();
    
    const message = userInput.value.trim();
    if (!message) return;
    
    // Clear input
    userInput.value = '';
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send message to server
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        addMessage(data.response, 'bot');
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage('Sorry, something went wrong. Please try again.', 'bot');
    }
}

/**
 * Send a quick message from the quick action buttons
 */
function sendQuickMessage(message) {
    userInput.value = message;
    chatForm.dispatchEvent(new Event('submit'));
}

/**
 * Add a message to the chat
 */
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'bot' ? '🤖' : '👤';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Handle multi-line responses
    const lines = text.split('\n');
    lines.forEach((line, index) => {
        if (line.trim()) {
            const p = document.createElement('p');
            p.textContent = line;
            content.appendChild(p);
        } else if (index > 0 && index < lines.length - 1) {
            // Add spacing for empty lines in the middle
            const br = document.createElement('br');
            content.appendChild(br);
        }
    });
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    // Remove quick actions after first user message
    const quickActions = chatMessages.querySelector('.quick-actions');
    if (quickActions && sender === 'user') {
        quickActions.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(indicator);
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Handle Enter key press
 */
userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Focus input on page load
window.addEventListener('load', function() {
    userInput.focus();
});
