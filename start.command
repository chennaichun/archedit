#!/bin/bash
# Double-click to start ControlNet Studio.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: installing packages..."
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
(sleep 2 && open http://localhost:8765) &
echo "ControlNet Studio is running at http://localhost:8765 (close this window to stop it)"
.venv/bin/uvicorn app.main:app --port 8765
