@echo off
echo 🚀 Starting Amazon ML Price Predictor...
echo ==========================================
echo.
echo 📊 Starting Flask Backend Server...
start cmd /c "cd python-backend && python app.py"
echo ✅ Backend starting on http://localhost:5000
echo.
echo ⏳ Waiting for backend to initialize...
timeout /t 3 /nobreak > nul
echo.
echo 🌐 Starting React Frontend...
start cmd /c "cd react-ml-app && npx react-scripts start"
echo ✅ Frontend starting on http://localhost:3000
echo.
echo ✅ Both services are starting!
echo 📱 Frontend: http://localhost:3000
echo ⚙️  Backend:  http://localhost:5000
echo 📚 API Docs: http://localhost:5000/api/health
echo.
echo 🎯 Your ML Prediction App is ready!
echo.
pause