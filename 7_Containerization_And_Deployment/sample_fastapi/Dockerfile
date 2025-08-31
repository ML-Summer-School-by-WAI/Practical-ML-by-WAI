FROM python:3.10.16


ENV PORT=8000


EXPOSE $PORT


WORKDIR /app
COPY . .

RUN pip install pipenv

RUN pipenv install --system --deploy


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]