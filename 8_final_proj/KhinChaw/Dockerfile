FROM tensorflow/tensorflow:2.14.0

WORKDIR /app
ENV PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        build-essential \
        hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages directly
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE ${PORT}
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]