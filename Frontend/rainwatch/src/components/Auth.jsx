import React, { useState } from 'react';
import '../styles.css'; // Create Auth.css

function Auth({ isLogin, isAdmin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [mobileNumber, setMobileNumber] = useState('');
  const [location, setLocation] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const endpoint = isAdmin ? '/api/admin/auth' : '/api/user/auth'; // Adjust endpoints
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password, mobileNumber, location }),
      });

      const data = await response.json();

      if (response.ok) {
        console.log(data.message); // Success message
        // Redirect or perform other actions upon successful auth
      } else {
        setError(data.message || 'Authentication failed.');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    }
  };

  return (
    <div className="auth-container">
      <h2>{isLogin ? (isAdmin ? 'Admin Login' : 'User Login') : (isAdmin ? 'Admin Register' : 'User Register')}</h2>
      {error && <p className="error">{error}</p>}
      <form onSubmit={handleSubmit}>
        <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} required />
        <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <input type="tel" placeholder="Mobile Number" value={mobileNumber} onChange={(e) => setMobileNumber(e.target.value)} required />
        <input type="text" placeholder="Location" value={location} onChange={(e) => setLocation(e.target.value)} required />
        <button type="submit">{isLogin ? 'Login' : 'Register'}</button>
      </form>
    </div>
  );
}

export default Auth;