import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import FloodMap from "./components/FloodMap"; // Import FloodMap Component
import Chatbot from "./components/Chatbot";
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
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
