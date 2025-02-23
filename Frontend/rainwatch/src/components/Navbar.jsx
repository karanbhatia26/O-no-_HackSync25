import React, { useState, useRef, useEffect } from "react";
import { FaBell, FaUser, FaUserShield } from "react-icons/fa";
import { Link } from "react-router-dom";
import "./Navbar.css"

const Navbar = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  const handleClickOutside = (event) => {
    if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
      setIsDropdownOpen(false);
    }
  };

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <nav className="navbar">
      <h2>RainWatch</h2>
      <div className="search-bar">
        <input type="text" placeholder="Search..." />
      </div>
      <div className="nav-icons" ref={dropdownRef}>
        <FaBell className="bell-icon" />

        <FaUserShield className="admin-icon" onClick={toggleDropdown} />
        {isDropdownOpen && (
          <div className="dropdown">
            <Link to="/user/login">User Login</Link>
            <Link to="/user/register">User Register</Link>
            <Link to="/admin/login">Admin Login</Link>
            <Link to="/admin/register">Admin Register</Link>
          </div>
        )}
        <img src="/profile.jpg" alt="Profile" />
      </div>
    </nav>
  );
};

export default Navbar;