FROM python:3.10.18-slim

ENV PORT=8000
EXPOSE $PORT

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipenv files first for caching
COPY Pipfile Pipfile.lock ./

# Install pipenv and Python dependencies
RUN pip install pipenv
RUN pipenv install --system --deploy
RUN pip install pydantic-core

# Copy your application code
COPY api_endpoint/ .

# Include your model file if needed by the app
COPY cats_dogs_mobilenet.keras .

# Start FastAPI with uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
