FROM python:3.11-slim

LABEL maintainer="nandini"
LABEL description="SQL Query Repair RL Environment — OpenEnv compliant"
LABEL version="1.0.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source
COPY . .

# HF Spaces runs on port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]