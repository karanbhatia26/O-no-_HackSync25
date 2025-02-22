import React from "react";
import { FaHome, FaMap, FaFlask, FaCog } from "react-icons/fa";

const Sidebar = () => {
  return (
    <aside className="sidebar">
      <ul>
        <li><FaHome /> Home</li>
        <li><FaMap /> Map</li>
        <li><FaFlask /> Ask</li>
        <li><FaCog /> Settings</li>
      </ul>
    </aside>
  );
};

export default Sidebar;
