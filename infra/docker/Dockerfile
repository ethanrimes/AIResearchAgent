FROM python:3.11

# Create a non-root user
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# Copy requirements and scripts
COPY infra/docker/requirements.txt .
COPY infra/docker/entrypoint.sh /app/infra/docker/entrypoint.sh
COPY infra/docker/setup_tests /app/infra/docker/setup_tests
COPY infra/docker/integration_tests /app/infra/docker/integration_tests

# Install dependencies
RUN pip install -r requirements.txt && \
    rm requirements.txt

# Make scripts executable
RUN chmod +x /app/infra/docker/entrypoint.sh && \
    chmod +x /app/infra/docker/setup_tests/run_tests.sh

# Create necessary directory structure with correct permissions
RUN mkdir -p /experiment /findings /hypothesis /planning /results /logs /scratchpad /read_only /data && \
    mkdir -p /app/core && \
    chown -R agent:agent /experiment /findings /hypothesis /planning /results /logs /scratchpad /read_only /data /app

# Use non-root user
USER agent

ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["/app/infra/docker/entrypoint.sh"]