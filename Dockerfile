# Retail Insights Agent: FastAPI (port 8000) and Streamlit (port 8501) in one container.
#   docker build -t retail-insights-agent .
#   docker run --rm -p 8501:8501 -p 8000:8000 -e GOOGLE_API_KEY=your-key retail-insights-agent
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
WORKDIR /app

COPY app/requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt -r requirements-api.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .

COPY app ./app
COPY scripts ./scripts
COPY data/docs ./data/docs
COPY data/deploy ./data/deploy
COPY data/examples ./data/examples
COPY .streamlit ./.streamlit

# Build the vector index at image build time (downloads the embedding model once). The DuckDB file is rebuilt from
# the parquet files on first start (a few seconds).
RUN python -m ria.ingest_docs

EXPOSE 8501 8000
CMD ["sh", "scripts/start.sh"]
