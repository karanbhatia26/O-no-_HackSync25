import React from "react";
import { FaBell } from "react-icons/fa";

const Navbar = () => {
  return (
    <nav className="navbar">
      <h2>RainWatch</h2>
      <div className="search-bar">
        <input type="text" placeholder="Search..." />
      </div>
      <div className="nav-icons">
        <FaBell className="bell-icon" />
        <img src="/profile.jpg" alt="Profile" />
      </div>
    </nav>
  );
};

export default Navbar;