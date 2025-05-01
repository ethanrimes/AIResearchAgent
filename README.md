
Bash commands

```
docker compose --env-file .env -f infra/docker-compose.yml config # show config
docker compose --env-file .env -f infra/docker-compose.yml up --build -d ai-agent # build container
docker compose rm -f # remove all containers 
```

