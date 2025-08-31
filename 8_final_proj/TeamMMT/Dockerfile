FROM python:3.11-slim

EXPOSE 8000

# Install minimal system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run FastAPI app with uvicorn using module syntax
CMD ["python", "-m", "uvicorn", "api_endpoint.main:app", "--host", "0.0.0.0", "--port", "8000"]
