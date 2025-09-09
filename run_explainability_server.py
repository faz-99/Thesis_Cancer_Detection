#!/usr/bin/env python3
"""
Run the explainability server with proper setup
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the environment for running the explainability server"""
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs/api", exist_ok=True)
    
    print("✅ Environment setup complete")

def run_server():
    """Run the FastAPI server"""
    
    setup_environment()
    
    print("🚀 Starting Explainability API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("🔬 Frontend available at: http://localhost:3000/explainability")
    print("\n" + "="*50)
    
    # Change to api directory and run server
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "explainability_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error running server: {e}")

if __name__ == "__main__":
    run_server()