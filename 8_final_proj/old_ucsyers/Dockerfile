FROM python:3.10.16-slim

WORKDIR /app

# Copy deps first for better build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app & model
COPY src/main.py .
COPY src/final_model.keras .

EXPOSE 8888
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
