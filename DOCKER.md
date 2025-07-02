# Docker Deployment Guide

This guide explains how to deploy the SubQL GraphQL Agent using Docker.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key

## Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd subql-graphql-agent
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the API**:
   - Server will be available at: `http://localhost:8000`
   - Health check: `http://localhost:8000/health`
   - API documentation: `http://localhost:8000/docs`

## Manual Docker Build

If you prefer to build manually:

```bash
# Build the image
docker build -t subql-graphql-agent .

# Run the container
docker run -d \
  --name subql-agent \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  -v $(pwd)/projects:/app/projects \
  subql-graphql-agent
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `LLM_MODEL` | LLM model to use | `gpt-4o-mini` |
| `IPFS_API_URL` | IPFS API endpoint | `https://unauthipfs.subquery.network/ipfs/api/v0` |
| `PORT` | Server port | `8000` |

## Data Persistence

Project data is stored in the `./projects` directory, which is mounted as a volume to persist data between container restarts.

## Health Check

The container includes a health check that monitors the `/health` endpoint. You can check the container health status:

```bash
docker ps
```

## Logs

View container logs:
```bash
docker-compose logs -f subql-graphql-agent
```

## Stopping the Service

```bash
docker-compose down
```

## Production Considerations

1. **Security**: Ensure your OpenAI API key is properly secured
2. **Reverse Proxy**: Consider using nginx or similar for production
3. **Resource Limits**: Set appropriate memory and CPU limits
4. **Backup**: Regularly backup the `projects/` directory
5. **Monitoring**: Set up proper monitoring and alerting

## Troubleshooting

- **Container won't start**: Check that your OpenAI API key is set correctly
- **Health check failing**: Ensure port 8000 is not blocked by firewall
- **Permission issues**: Ensure the `projects/` directory is writable

For more detailed logs:
```bash
docker-compose logs subql-graphql-agent
```