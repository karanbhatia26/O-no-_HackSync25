# RainWatch - Mumbai Flood Evacuation System

An intelligent flood evacuation system that helps Mumbai residents find safe routes during flood emergencies using ML-powered route optimization and real-time risk assessment.

## 🌟 Features

- **Dynamic Evacuation Routes**: ML-powered route generation avoiding flood-prone areas
- **Real-time Risk Assessment**: Zone-based risk visualization (Red, Yellow, Blue)
- **Interactive Maps**: User-friendly interface showing safe paths and danger zones
- **Multi-point Navigation**: Support for multiple starting locations
- **Emergency Support**: Quick access to emergency contacts and safety guidelines

## 🛠️ Tech Stack

- **Frontend**: React + Vite
- **Backend**: Python with FastAPI
- **ML Model**: PyTorch
- **Maps**: Leaflet + Folium
- **Graph Processing**: OSMNX, NetworkX
- **Visualization**: Folium

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- Git
- CUDA-compatible GPU (recommended)

## 🚀 Installation

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rainwatch.git
cd rainwatch

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd Frontend/rainwatch

# Install Node dependencies
npm install

# Start development server
npm run dev
```

### ML Model Setup

```bash
# Navigate to ML directory
cd ml

# Run model training (optional)
python ConnectedRoads3.py

# Test route generation
python test.py
```

## 💻 Usage

1. **Start the Backend Server**:
```bash
cd flask_server/api
uvicorn models:app --reload
```

2. **Launch the Frontend**:
```bash
cd Frontend/rainwatch
npm run dev
```

3. **Access the Application**:
- Open `http://localhost:5173` in your browser
- Select your location on the map
- View generated evacuation routes
- Follow safety guidelines

## 🗺️ Route Generation

The system uses three zone types:
- 🔴 **Red Zones**: High-risk flood-prone areas
- 🟡 **Yellow Zones**: Medium-risk areas
- 🔵 **Blue Zones**: Safe zones/evacuation points

Routes are optimized for:
- Minimal exposure to flood-prone areas
- Shortest safe path to evacuation points
- Real-time traffic conditions
- Bridge and elevated road preferences

## 🔧 Configuration

Create a `.env` file in the project root:

```env
VITE_API_URL=http://localhost:8000
VITE_MAPS_API_KEY=your_maps_api_key
```

## 📖 API Documentation

### GET `/api/evacuation-route`
Returns evacuation route HTML for a given location

**Parameters**:
```json
{
  "lat": 19.2356,
  "lon": 73.1292,
  "location_name": "Kalyan_Station"
}
```

**Response**: HTML file containing route visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 👥 Team

- Karan Bhatia
- Nickhil Shivakumar
- Kushl Alve
- V