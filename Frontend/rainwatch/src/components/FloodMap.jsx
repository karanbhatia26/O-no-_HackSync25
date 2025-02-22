import React from 'react';

function FloodMap() {
  return (
    <div className="flood-map-content">
      <div className="map-container">
        <div className="interactive-map">
          Interactive Map
        </div>
        <div className="risk-levels">
          <h3>Risk Levels</h3>
          <div className="level safe">Safe</div>
          <div className="level moderate">Moderate</div>
          <div className="level high">High Risk</div>
        </div>
      </div>
      <div className="data-insights">
        <div className="data-header">
          Data Insights
        </div>
        <div className="data-info">
          <p>River Level: 3.5m</p>
          <p>Rainfall: 120mm</p>
          <p>Flood Risk: Medium</p>
        </div>
        <div className="data-forecast">
          Day Forecast...
        </div>
      </div>
    </div>
  );
}

export default FloodMap;