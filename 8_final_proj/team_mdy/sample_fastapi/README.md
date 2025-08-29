### docker build and Run commands
```bash
# build
docker build . -t test_tt


# docker run 
docker run -p 8000:8000 --env PORT=8000 test_tt

# List running containers.
docker ps 


# Stop a running container.
docker stop <container_id>

# Start a stopped container.
docker start <container_id>

# Remove a container.
docker rm <container_id>

```

### Clean up commands
```bash

# Remove unused containers, images, volumes, and networks.
docker system prune

# Remove unused images.
docker image prune

# Remove stopped containers.
docker container prune 

#Remove unused volumes.
docker volume prune
```
