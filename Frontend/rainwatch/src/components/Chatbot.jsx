import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import '../styles.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [city, setCity] = useState('Mumbai');

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      const response = await fetch(
        `http://localhost:8000/chat?city=${encodeURIComponent(city)}&user_message=${encodeURIComponent(input)}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const data = await response.json();
      
      // Format and combine all responses
      let formattedResponse = '';
      
      // Add flood warning if probability is high
      if (data.flood_probability > 0.7) {
        formattedResponse += `## ⚠️ Warning\n**High flood probability (${(data.flood_probability * 100).toFixed(1)}%) in ${data.city}**\n\n`;
      }

      // Add weather info
      formattedResponse += `### Current Weather in ${data.city}\n`;
      if (data.weather) {
        formattedResponse += `- Temperature: ${data.weather.temperature}°C\n`;
        formattedResponse += `- Humidity: ${data.weather.humidity}%\n`;
        formattedResponse += `- Wind Speed: ${data.weather.wind_speed}m/s\n\n`;
      }

      // Add chat responses
      if (data.chat_response.chat_response) {
        Object.values(data.chat_response.chat_response).forEach(response => {
          formattedResponse += `${response}\n`;
        });
      }

      setMessages(prev => [...prev, {
        text: formattedResponse,
        sender: 'bot',
        isMarkdown: true
      }]);

    } catch (error) {
      console.error('Error fetching chatbot response:', error);
      setMessages(prev => [...prev, { 
        text: '### Error\nSorry, something went wrong. Please try again.',
        sender: 'bot',
        isMarkdown: true
      }]);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chat-header">
        <h3>RainWatch Assistant</h3>
        <p>Current city: {city}</p>
      </div>
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.isMarkdown ? (
              <ReactMarkdown>{message.text}</ReactMarkdown>
            ) : (
              message.text
            )}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Ask about weather or flood precautions..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
}

export default Chatbot;