FROM python:3.11

# Create a non-root user to run the container
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# --------------------------------------------
# STEP 1: Copy and install dependencies FIRST
# This is cached if requirements.txt doesn’t change
# --------------------------------------------
COPY infra/docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# --------------------------------------------
# STEP 2: Copy the rest of the app (after pip)
# This avoids invalidating the previous layer unnecessarily
# --------------------------------------------
COPY . /app

# Copy entrypoint script explicitly (already included by COPY .)
# But still make it executable
RUN chmod +x /app/infra/docker/entrypoint.sh

# Create writable and read-only sandbox dirs
RUN mkdir -p /app/sandbox/writable /app/sandbox/read_only \
    && chown -R agent:agent /app/sandbox/writable

# Switch to non-root user
USER agent

# Set entrypoint
ENTRYPOINT ["/app/infra/docker/entrypoint.sh"]