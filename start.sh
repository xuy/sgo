#!/bin/bash
# Kill any process holding port 7860
kill $(lsof -t -i:7860 2>/dev/null) 2>/dev/null || true
sleep 1
exec python -u web/app.py
