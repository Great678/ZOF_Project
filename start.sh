#!/bin/bash

# ZOF Solver - Quick Start Script

echo "======================================"
echo "   ZOF Solver - Quick Start"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✓ pip3 found"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
    echo ""
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Menu
echo "======================================"
echo "What would you like to do?"
echo "======================================"
echo "1. Run CLI Application"
echo "2. Run Web Application"
echo "3. Run Tests"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting CLI Application..."
        echo ""
        python3 ZOF_CLI.py
        ;;
    2)
        echo ""
        echo "🚀 Starting Web Application..."
        echo "📱 Open your browser to: http://localhost:5000"
        echo "⏹️  Press Ctrl+C to stop the server"
        echo ""
        python3 app.py
        ;;
    3)
        echo ""
        echo "🧪 Running Tests..."
        echo ""
        python3 test_solver.py
        ;;
    4)
        echo "Goodbye! 👋"
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac
