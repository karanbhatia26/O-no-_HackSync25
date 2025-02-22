import React from "react";
import { FaHome, FaMap, FaFlask, FaCog } from "react-icons/fa";
import { Link } from "react-router-dom";
const Sidebar = () => {
  return (
    <aside className="sidebar">
      <ul>
        <li><FaHome /> Home</li>
        <li>
          <Link to="/floodmap"> {/* Use Link to route to /floodmap */}
            <FaMap /> Map
          </Link>
        </li>
        <li><FaFlask /> Ask</li>
        <li><FaCog /> Settings</li>
      </ul>
    </aside>
  );
};

export default Sidebar;