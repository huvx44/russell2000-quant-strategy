FROM python:3.11-slim

# System dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY webapp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY russell2000_backtest.py /app/core/russell2000_backtest.py
COPY smallcap_backtest.py /app/core/smallcap_backtest.py
COPY portfolio_manager.py /app/core/portfolio_manager.py
COPY russell2000_tickers.csv /app/core/russell2000_tickers.csv
COPY russell2000_tickers.txt /app/core/russell2000_tickers.txt
COPY webapp/ /app/webapp/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV CORE_DIR=/app/core

# Create data directory
RUN mkdir -p /app/data

EXPOSE 5000

WORKDIR /app/webapp

CMD ["gunicorn", \
     "--worker-class=gthread", \
     "--workers=1", \
     "--threads=4", \
     "--bind=0.0.0.0:5000", \
     "--timeout=600", \
     "wsgi:app"]
