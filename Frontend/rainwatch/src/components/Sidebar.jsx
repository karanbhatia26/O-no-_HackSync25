import React from "react";
import { FaHome, FaMap, FaFlask, FaCog } from "react-icons/fa";
import { Link } from "react-router-dom";
import "./Sidebar.css";

const Sidebar = () => {
  return (
    <aside className="sidebar">
      <ul>
        <li>
        <Link to="/"><FaHome /> Home
        </Link>
        </li>
        <li>
          <Link to="/floodmap"> {/* Use Link to route to /floodmap */}
            <FaMap /> Map
          </Link>
        </li>
        <li>
          <Link to="/chatbot"> {/* Link to Chatbot */}
            <FaFlask /> Ask
          </Link>
        </li>
        <li><FaCog /> Settings</li>
      </ul>
    </aside>
  );
};

export default Sidebar;