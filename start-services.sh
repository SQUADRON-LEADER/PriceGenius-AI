#!/bin/bash
# Amazon ML Price Predictor - Startup Script

echo "🚀 Starting Amazon ML Price Predictor..."
echo "=================================="
echo ""

# Start Flask Backend
echo "📊 Starting Flask Backend Server..."
cd python-backend
start cmd /k "python app.py"
cd ..

# Wait for backend to initialize
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Start React Frontend
echo "🌐 Starting React Frontend..."
cd react-ml-app
start cmd /k "npx react-scripts start"
cd ..

echo ""
echo "✅ Both services are starting!"
echo "📱 Frontend: http://localhost:3000"
echo "⚙️  Backend:  http://localhost:5000"
echo "📚 API Docs: http://localhost:5000/api/health"
echo ""
echo "Press any key to exit..."
read -n1