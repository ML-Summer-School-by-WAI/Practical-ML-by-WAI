FROM python:3.10-slim

EXPOSE 8000

COPY src/ ./src

COPY Pipfile Pipfile.lock ./

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv && pipenv install --system --deploy --ignore-pipfile

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]