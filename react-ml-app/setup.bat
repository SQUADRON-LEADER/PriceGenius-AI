@echo off
echo ========================================
echo   Amazon ML Price Predictor Setup
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js is installed
node --version

REM Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed. Please install npm with Node.js
    pause
    exit /b 1
)

echo ✅ npm is installed
npm --version
echo.

echo 📦 Installing dependencies...
echo This may take a few minutes...
echo.

REM Install dependencies
npm install

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.
echo 🚀 Starting the development server...
echo.
echo The application will open in your browser at:
echo http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the development server
npm start

pause