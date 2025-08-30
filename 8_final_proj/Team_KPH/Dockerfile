FROM python:3.11.2-slim

ENV PORT=8000

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip pipenv

RUN pipenv install --system --deploy

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]