FROM python:3.10.16


ENV PORT=8000


EXPOSE $PORT


WORKDIR /app

COPY . .

RUN pip install pipenv

RUN pipenv install --system --deploy

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]