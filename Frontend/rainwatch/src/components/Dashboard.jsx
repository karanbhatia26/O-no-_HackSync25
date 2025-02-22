import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const data = [
  { name: "Jan", rainfall: 30 },
  { name: "Feb", rainfall: 20 },
  { name: "Mar", rainfall: 50 },
  { name: "Apr", rainfall: 80 },
  { name: "May", rainfall: 60 },
  { name: "Jun", rainfall: 90 },
];

const Dashboard = () => {
  return (
    <div className="dashboard">
      <h2>Risk Level: <span className="safe">Safe</span></h2>
      <div className="grid">
        <div className="card">Rainfall <br /> <strong>2.73 mm</strong></div>
        <div className="card">Humidity <br /> <strong>60%</strong></div>
        <div className="card">Temperature <br /> <strong>284.2 K</strong></div>
        <div className="card">Wind Speed <br /> <strong>4.09 m/s</strong></div>
        <div className="card">Cloud Cover <br /> <strong>83%</strong></div>

        {/* Graph */}
        <div className="chart">
          <h3>Rainfall vs Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="rainfall" stroke="#0072ff" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;