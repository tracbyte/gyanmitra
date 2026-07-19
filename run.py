"""
GyanMitra — Application Entry Point
Run with:  python run.py
Production: gunicorn -w 4 run:app
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from api.app import app

if __name__ == "__main__":
    print("\n🎓 GyanMitra IVA Backend")
    print("   API:      http://localhost:5050/api/health")
    print("   Chat:     POST http://localhost:5050/api/chat")
    print("   Escalations: http://localhost:5050/api/escalations\n")
    app.run(debug=True, port=5050, host="0.0.0.0")
