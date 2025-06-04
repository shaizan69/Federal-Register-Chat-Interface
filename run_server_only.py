import uvicorn
import os
import sys

# Add the current directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("Starting FastAPI server without data pipeline...")
    print("Access the web interface at http://localhost:8000")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # Using localhost for Windows
        port=8000,
        reload=True
    ) 