FROM python:3.11-slim

# Metadata
LABEL maintainer="nandini"
LABEL description="SQL Query Repair RL Environment — OpenEnv compliant"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces runs on port 7860
EXPOSE 7860

# Environment defaults (overridable at runtime)
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check — lets Docker and HF Spaces know when the server is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Use server/app.py main() — satisfies OpenEnv multi-mode deployment requirement
CMD ["python", "server/app.py"]