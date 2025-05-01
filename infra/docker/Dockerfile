FROM python:3.11

# Create a non-root user
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# --- Step 1: Copy only requirements and install dependencies ---
COPY infra/docker/requirements.txt .
RUN pip install -r requirements.txt

# --- Step 2: Copy everything else (after deps installed) ---
# This layer will only invalidate if app code changes
COPY core /app/core
COPY infra/docker/entrypoint.sh /app/infra/docker/entrypoint.sh
COPY infra/docker/setup_tests /app/infra/docker/setup_tests
COPY sandbox /app/sandbox
COPY data /app/data
COPY README.md /app/README.md

# --- Make necessary scripts executable ---
RUN chmod +x /app/infra/docker/entrypoint.sh \
 && chmod +x /app/infra/docker/setup_tests/run_tests.sh

# --- Create sandbox dirs & set permissions ---
RUN mkdir -p /app/sandbox/writable /app/sandbox/read_only \
 && chown -R agent:agent /app/sandbox/writable

# --- Use non-root user ---
USER agent

ENV PYTHONPATH=/app

# --- Set entrypoint ---
ENTRYPOINT ["/app/infra/docker/entrypoint.sh"]