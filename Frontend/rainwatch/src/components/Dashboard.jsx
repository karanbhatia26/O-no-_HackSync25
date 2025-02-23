import React, { useState, useEffect } from "react";

const Dashboard = () => {
  const [weather, setWeather] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const location = "London";

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        // Fetch Weather Data
        const weatherResponse = await fetch(`http://127.0.0.1:8000/weather?city=${location}`);
        if (!weatherResponse.ok) throw new Error(`Weather API error: ${weatherResponse.status}`);
        const weatherData = await weatherResponse.json();

        // Fetch Prediction Data
        const predictResponse = await fetch(`http://127.0.0.1:8000/predict?city=${location}`);
        if (!predictResponse.ok) throw new Error(`Predict API error: ${predictResponse.status}`);
        const predictData = await predictResponse.json();

        console.log("Fetched weather data:", weatherData);
        console.log("Fetched prediction data:", predictData);

        setWeather(weatherData);
        setPrediction(predictData); // Extracting only the weather part of predict API response
      } catch (error) {
        console.error("Error fetching data:", error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchWeather();
  }, []);

  if (loading) return <p>Loading weather data...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <div className="dashboard">
      <h2>Weather Dashboard</h2>
      <div className="grid">
        {/* Weather API Data */}
        <div className="card">Current Temperature: <strong>{weather.main.temp}°C</strong></div>
        <div className="card">Min Temp: <strong>{weather.main.temp_min}°C</strong></div>
        <div className="card">Max Temp: <strong>{weather.main.temp_max}°C</strong></div>
        <div className="card">Pressure: <strong>{weather.main.pressure} hPa</strong></div>
        <div className="card">Humidity: <strong>{weather.main.humidity}%</strong></div>
        <div className="card">Wind Speed: <strong>{weather.wind.speed} m/s</strong></div>

        {/* Prediction API Data */}
        {prediction && (
          <>
            <div className="card">Cloud Coverage: <strong>{prediction.weather.cloud_coverage}%</strong></div>
            <div className="card">Rainfall: <strong>{prediction.weather.rainfall} mm</strong></div>
            <div className="card">Flood Probability: <strong>{(prediction.flood_probability * 100).toFixed(2)}%</strong></div>
          </>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
