FROM python:3.11

# Create a non-root user to run the container
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# Copy requirements file
COPY infra/docker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint script
COPY infra/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create necessary directories for the container
RUN mkdir -p /app/sandbox/writable /app/sandbox/read_only

# Set proper permissions
RUN chown -R agent:agent /app/sandbox/writable

# Switch to non-root user
USER agent

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]