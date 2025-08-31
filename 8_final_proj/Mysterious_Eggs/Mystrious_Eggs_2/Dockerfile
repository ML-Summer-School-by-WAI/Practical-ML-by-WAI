FROM python:3.12-slim

WORKDIR /app


RUN pip install --upgrade pip && pip install pipenv

COPY Pipfile ./

RUN pipenv install

COPY . .

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:5001", "app:app"]