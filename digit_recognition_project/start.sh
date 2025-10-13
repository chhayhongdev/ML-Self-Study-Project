#!/usr/bin/env bash
set -euo pipefail

# Start Digit Recognition API
# Usage: ./start.sh [command]
# Commands: start (default), stop, restart

MODE=${1:-start}

case "$MODE" in
    "start")
        echo "🚀 Starting Digit Recognition API on port 8000..."
        docker-compose up -d
        echo "📍 API available at: http://localhost:8000"
        echo "🌐 Web interface at: http://localhost:8000/web_interface/index.html"
        echo "🔍 Health check at: http://localhost:8000/health"
        ;;
    "stop")
        echo "� Stopping Digit Recognition API..."
        docker-compose down
        ;;
    "restart")
        echo "� Restarting Digit Recognition API..."
        docker-compose restart
        ;;
    *)
        echo "Usage: $0 [start|stop|restart]"
        echo "  start   - Start the API (default)"
        echo "  stop    - Stop the API"
        echo "  restart - Restart the API"
        exit 1
        ;;
esac