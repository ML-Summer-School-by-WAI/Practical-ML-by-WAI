# Use official Python 3.11 slim image
FROM python:3.11-slim

EXPOSE 8000

WORKDIR /app
COPY . .

RUN pip install pipenv

RUN pipenv install --system --deploy

# Run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
