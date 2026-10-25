#!/bin/sh
# Starts the FastAPI app and the Streamlit app together (used by the Docker image).
# Streamlit talks to the API over HTTP (RIA_API_URL), so both go through the same usage limits.
set -e
python -c "from ria.bootstrap import ensure_data; print('data ready:', ensure_data() or 'already present')"
uvicorn ria.api:app --host 0.0.0.0 --port "${API_PORT:-8000}" &
export RIA_API_URL="http://127.0.0.1:${API_PORT:-8000}"
exec streamlit run app/streamlit_app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0
