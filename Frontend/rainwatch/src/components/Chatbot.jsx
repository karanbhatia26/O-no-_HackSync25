import React, { useState } from 'react';
import '../styles.css'; // Create a Chatbot.css file

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    // Add user message to the chat
    setMessages([...messages, { text: input, sender: 'user' }]);

    // Clear input
    setInput('');

    // Simulate API call (replace with your actual API call)
    try {
      const response = await fetch('/api/chatbot', { // Replace with your API endpoint
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();
      setMessages([...messages, { text: data.response, sender: 'bot' }]); // Add bot response
    } catch (error) {
      console.error('Error fetching chatbot response:', error);
      setMessages([...messages, { text: 'Sorry, something went wrong.', sender: 'bot' }]);
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.text}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          placeholder="Type your message..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
}

export default Chatbot;