import { useNavigate } from "react-router-dom";
import "./UserAdminSelection.css";
import React from "react";
import "../assets/rainwatch_logo.png"

export default function UserAdminSelection() {              
    const navigate = useNavigate();
    return (
      <div className="selection-container">
        <h1>Welcome to <span>RainWatch</span>!</h1>
        <h2>Select your role</h2>
        <div className="buttons">
            <button onClick={() => navigate("/user/register")}>User</button>
            <button onClick={() => navigate("/admin/register")}>Admin</button>
        </div>
      </div>
    )
}