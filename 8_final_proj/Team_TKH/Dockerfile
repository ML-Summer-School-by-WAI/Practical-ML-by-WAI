# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY Pipfile /app/

# Install dependencies using pip directly
RUN pip install --no-cache-dir fastapi uvicorn python-multipart opencv-python numpy pillow tqdm jupyter matplotlib tensorflow==2.19.0 pydot graphviz

# Copy rest of the project
COPY . /app/

# Expose FastAPI default port (customizable)
EXPOSE 8888

# Run FastAPI with uvicorn
CMD ["python","-m","uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8888"]
