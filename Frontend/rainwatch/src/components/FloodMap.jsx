import React from "react";
import "../styles.css";

const FloodMap = () => {
  return (
    <main className="flood-map">
      {/* Map Section */}
      <section className="map-section">
        <h3>Interactive Map</h3>
        <div className="map-container">
          {/* Placeholder for Interactive Map (Use Leaflet.js or Google Maps API) */}
          <p>Interactive Map</p>
        </div>
        {/* Risk Level Legend */}
        <div className="legend">
          <p>Risk Levels:</p>
          <div className="legend-item"><span className="safe"></span> Safe</div>
          <div className="legend-item"><span className="moderate"></span> Moderate</div>
          <div className="legend-item"><span className="high-risk"></span> High Risk</div>
        </div>
      </section>

      {/* Data Insights */}
      <aside className="data-insights">
        <h3>Data Insights</h3>
        <p><strong>River Level:</strong> 3.5m</p>
        <p><strong>Rainfall:</strong> 120mm</p>
        <p><strong>Flood Risk:</strong> Medium</p>

        {/* 8-Day Forecast */}
        <div className="forecast">
          <h4>8-Day Forecast</h4>
          <ul>
            <li>Sat, Feb 22 - 12°C/7°C - Moderate Rain</li>
            <li>Sun, Feb 23 - 15°C/9°C - Light Rain</li>
            <li>Mon, Feb 24 - 16°C/10°C - Cloudy</li>
            <li>Tue, Feb 25 - 17°C/11°C - Clear Sky</li>
            <li>Wed, Feb 26 - 10°C/6°C - Cloudy</li>
            <li>Thu, Feb 27 - 9°C/5°C - Light Rain</li>
            <li>Fri, Feb 28 - 8°C/4°C - Light Rain</li>
            <li>Sat, Mar 01 - 9°C/6°C - Slight Rain</li>
          </ul>
        </div>
      </aside>
    </main>
  );
};

export default FloodMap;
