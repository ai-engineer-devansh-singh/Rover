# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Install runtime dependencies: git for metrics, sqlite3 for Cursor parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN pip install -e . --no-deps

# Signal to config.py that we're inside Docker
ENV ROVER_IN_DOCKER=1

# Default mounts expected:
#   ~/.claude:/claude:ro
#   ~/.codex:/codex:ro
#   ~/.config/Cursor:/cursor:ro
#   $(pwd):/workspace:ro     (repo for git metrics)
#   $(pwd):/reports           (writable output dir)

WORKDIR /workspace
ENTRYPOINT ["rover"]
CMD ["analyze", "--since", "2m", "--output", "/reports/rover-report.html"]
