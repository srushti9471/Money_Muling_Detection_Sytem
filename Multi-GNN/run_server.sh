#!/usr/bin/env bash
# ==============================================================================
# MONEY MULING DETECTION SYSTEM - Server Launcher & Port Manager
# Usage:
#   ./run_server.sh         (starts on default port 8000)
#   ./run_server.sh 8080    (starts on custom port 8080)
# ==============================================================================

set -e

PORT="${1:-8000}"
HOST="0.0.0.0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

echo "=========================================================="
echo "🛡️  MONEY MULING DETECTION SYSTEM"
echo "=========================================================="
echo "Checking port $PORT for existing processes..."

# 1. Find any process currently listening on the port
PIDS=$(lsof -t -i :"$PORT" 2>/dev/null || true)

if [ -n "$PIDS" ]; then
    echo "⚠️  Port $PORT is occupied by PID(s): $PIDS"
    echo "Terminating conflicting process(es)..."
    
    # Attempt graceful shutdown first
    kill -15 $PIDS 2>/dev/null || true
    sleep 1

    # Force kill if still lingering
    REMAINING=$(lsof -t -i :"$PORT" 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        echo "Force killing remaining process(es): $REMAINING"
        kill -9 $REMAINING 2>/dev/null || true
        sleep 1
    fi

    # Extra fallback using fuser
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 0.5
    echo "✅ Port $PORT successfully cleared."
else
    echo "✅ Port $PORT is free."
fi

# 2. Verify port is completely available
if lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "❌ Error: Port $PORT is still in use. Please check running processes manually."
    exit 1
fi

# 3. Check for required checkpoint file
if [ ! -f "checkpoints/detected_aml_alerts.csv" ]; then
    echo "⚠️  Warning: checkpoints/detected_aml_alerts.csv not found."
    echo "The server requires this file for pre-computed graph evaluation."
fi

echo "----------------------------------------------------------"
echo "🚀 Starting FastAPI server on http://localhost:${PORT}"
echo "----------------------------------------------------------"
echo "Press CTRL+C to stop the server at any time."
echo ""

# 4. Exec uvicorn server
exec python3 -m uvicorn server:app --host "$HOST" --port "$PORT"
