FROM python:3.12-slim

# System deps for unstructured (poppler, tesseract), opencv, and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/
COPY scripts/ scripts/
COPY .env .env

# Create data directories
RUN mkdir -p data/raw data/processed data/indices

# Create .streamlit config
RUN mkdir -p .streamlit
RUN echo '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\n\n[theme]\nprimaryColor = "#E91E8C"\nbackgroundColor = "#FFF5F7"\nsecondaryBackgroundColor = "#FFE4EC"\ntextColor = "#333333"' > .streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Default: run the Streamlit app
CMD ["streamlit", "run", "src/app/main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
