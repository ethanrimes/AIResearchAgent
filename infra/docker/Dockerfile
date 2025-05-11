FROM python:3.11

# Create a non-root user
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY infra/docker/requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY core /app/core
COPY infra/docker/entrypoint.sh /app/infra/docker/entrypoint.sh
COPY infra/docker/setup_tests /app/infra/docker/setup_tests
COPY sandbox/read_only /app/sandbox/read_only
COPY data /app/data
COPY README.md /app/README.md

# Make scripts executable
RUN chmod +x /app/infra/docker/entrypoint.sh \
 && chmod +x /app/infra/docker/setup_tests/run_tests.sh

# Create directories for flattened structure
RUN mkdir -p /experiment /findings /hypothesis /planning /results \
 && chown -R agent:agent /experiment /findings /hypothesis /planning /results \
 && mkdir -p /read_only \
 && chown -R agent:agent /read_only

# Create sandbox dirs for backward compatibility
RUN mkdir -p /app/sandbox/writable \
 && chown -R agent:agent /app/sandbox/writable

# Use non-root user
USER agent

ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["/app/infra/docker/entrypoint.sh"]