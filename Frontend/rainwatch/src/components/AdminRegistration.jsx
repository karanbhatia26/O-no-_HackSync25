import React, { useState } from "react";
import "./AdminRegistration.css";
import { Link } from "react-router-dom";

function AdminRegister() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [mobileNumber, setMobileNumber] = useState("");
  const [location, setLocation] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const response = await fetch("/api/admin/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, mobileNumber, location }),
      });

      const data = await response.json();

      if (response.ok) {
        console.log("Admin registered successfully!");
      } else {
        setError(data.message || "Registration failed.");
      }
    } catch (err) {
      setError("An error occurred. Please try again.");
    }
  };

  return (
    <div className="auth-page">
        <div className="auth-container">
          <h2>Admin Registration</h2>
          {error && <p className="error">{error}</p>}
          <form onSubmit={handleSubmit}>
            <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} required />
            <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            <input type="tel" placeholder="Mobile Number" value={mobileNumber} onChange={(e) => setMobileNumber(e.target.value)} required />
            <input type="text" placeholder="Location" value={location} onChange={(e) => setLocation(e.target.value)} required />
            <button type="submit">Register</button>
          </form>
          <p>Already have an account? <Link to="/admin/login">Login here</Link></p>
        </div>
    </div>
  );
}

export default AdminRegister;
