import React, { useEffect, useState } from 'react';

function FloodMap() {
  const [mapHtml, setMapHtml] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Use the correct path to the public directory
    fetch('../evacuation_route.html')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to load map');
        }
        return response.text();
      })
      .then(html => {
        // Add required Leaflet scripts and styles
        const enhancedHtml = `
          <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
          <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
          <style>
            html, body { margin: 0; padding: 0; height: 100%; }
            #map { width: 100%; height: 100%; }
          </style>
          ${html}
        `;
        setMapHtml(enhancedHtml);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading map:', error);
        setError(error.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="loading">Loading map...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="flood-map-content">
      <div className="map-container">
        <div className="interactive-map">
          <iframe
            srcDoc={mapHtml}
            style={{
              width: '100%',
              height: '70vh', // Changed to viewport height for better responsiveness
              minHeight: '600px',
              border: 'none',
              borderRadius: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            title="Evacuation Route Map"
            sandbox="allow-scripts allow-same-origin"
          />
        </div>
        <div className="risk-levels">
          <h3>Risk Levels</h3>
          <div className="level">
            <span className="dot blue"></span>
            Safe Zone
          </div>
          <div className="level">
            <span className="dot yellow"></span>
            Moderate Risk
          </div>
          <div className="level">
            <span className="dot red"></span>
            High Risk
          </div>
        </div>
      </div>
    </div>
  );
}

export default FloodMap;