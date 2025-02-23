import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState } from "react";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import FloodMap from "./components/FloodMap"; // Import FloodMap Component
import Chatbot from "./components/Chatbot";
import { UserLogin, UserRegister, AdminLogin, AdminRegister } from "./components/AuthComponents";
import UserAdminSelection from "./components/UserAdminSelection.jsx";
import "./styles.css";
import { Navigate } from "react-router-dom";
import { useLocation } from "react-router-dom";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false); // Track login status

  return (
    <Router>
      <div className="container">
        <ConditionalNavbar isAuthenticated={isAuthenticated} />
        <div className="main-content">
          <ConditionalSidebar isAuthenticated={isAuthenticated} />
          <Routes>
            {/* Step 1: Selection Page */}
            <Route path="/" element={<UserAdminSelection />} />
            
            {/* Step 2: Authentication Routes */}
            <Route path="/user/register" element={<UserRegister setAuth={setIsAuthenticated} />} />
            <Route path="/user/login" element={<UserLogin setAuth={setIsAuthenticated} />} />
            <Route path="/admin/register" element={<AdminRegister setAuth={setIsAuthenticated} />} />
            <Route path="/admin/login" element={<AdminLogin setAuth={setIsAuthenticated} />} />

            {/* Step 3: Protected Routes */}
            <Route path="/dashboard" element={isAuthenticated ? <Dashboard /> : <Navigate to="/" />} />
            <Route path="/floodmap" element={isAuthenticated ? <FloodMap /> : <Navigate to="/" />} />
            <Route path="/chatbot" element={isAuthenticated ? <Chatbot /> : <Navigate to="/" />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

// Navbar should only show when authenticated
const ConditionalNavbar = ({ isAuthenticated }) => {
  const location = useLocation();
  const authPaths = ["/", "/user/login", "/user/register", "/admin/login", "/admin/register"];
  return !authPaths.includes(location.pathname) && isAuthenticated ? <Navbar /> : null;
};

// Sidebar should only show when authenticated
const ConditionalSidebar = ({ isAuthenticated }) => {
  const location = useLocation();
  const authPaths = ["/", "/user/login", "/user/register", "/admin/login", "/admin/register"];
  return !authPaths.includes(location.pathname) && isAuthenticated ? <Sidebar /> : null;
};

export default App;
