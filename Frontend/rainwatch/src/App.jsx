import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import FloodMap from "./components/FloodMap"; // Import FloodMap Component
import Chatbot from "./components/Chatbot";
import { UserLogin, UserRegister, AdminLogin, AdminRegister } from "./components/AuthComponents";
import "./styles.css";

function App() {
 
  
  return (
    <Router>
      <div className="container">
        <Navbar />
        <div className="main-content">
          <Sidebar />
          <Routes>
            <Route path="/" element={<Dashboard />} /> {/* Home Page (Dashboard) */}
            <Route path="/floodmap" element={<FloodMap />} /> {/* Flood Map Page */}
            <Route path="/chatbot" element={<Chatbot />} /> {/* Add Chatbot route */}
            <Route path="/user/login" element={<UserLogin />} />
            <Route path="/user/register" element={<UserRegister />} />
            <Route path="/admin/login" element={<AdminLogin />} />
            <Route path="/admin/register" element={<AdminRegister />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
