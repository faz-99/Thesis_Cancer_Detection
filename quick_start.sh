#!/bin/bash

echo "🚀 Breast Cancer Explainability - Quick Start"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi

echo "✅ Python and Node.js found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install torch torchvision opencv-python scikit-image matplotlib numpy Pillow scipy fastapi uvicorn python-multipart

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads outputs/api models

# Check if model exists
if [ ! -f "models/efficientnet_b0_best.pth" ]; then
    echo "⚠️  No trained model found. Will use demo model."
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
if [ -f "package.json" ]; then
    npm install
else
    echo "⚠️  Frontend package.json not found"
fi
cd ..

echo ""
echo "🎯 Ready to start!"
echo ""
echo "Run these commands in separate terminals:"
echo ""
echo "Terminal 1 (Backend):"
echo "python run_explainability_server.py"
echo ""
echo "Terminal 2 (Frontend):"
echo "cd frontend && npm run dev"
echo ""
echo "Then visit: http://localhost:3000/explainability"
echo ""
echo "Or run demo: python demo_explainability.py"