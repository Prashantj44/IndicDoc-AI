FROM python:3.11-slim

LABEL maintainer="IndicDoc AI" \
      description="Deep Learning-Based Multilingual Indian Document Layout Understanding"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (not data/outputs)
COPY ml/ ml/
COPY app/ app/
COPY docs/ docs/
COPY tests/ tests/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY README.md .

# Create output directories
RUN mkdir -p data/processed/sample data/raw outputs/models outputs/metrics outputs/plots outputs/predictions/errors

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
