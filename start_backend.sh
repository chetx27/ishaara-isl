#!/bin/bash

# Start the Ishaara backend server locally

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the FastAPI server with Uvicorn
echo "Starting Ishaara Backend API..."
echo "Server will be available at: http://localhost:8000"
echo "API Docs at: http://localhost:8000/docs"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
