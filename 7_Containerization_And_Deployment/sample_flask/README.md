
#### Docker Run
```bash
docker build . -t test_flask
docker run -p 3000:8000 test_flask

docker run -p 3000:8000 --env PORT=8000 test_flask


```