# Dockerfile for GoatBotTrade Backend
# ================================

# Use slim python image for efficiency
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# Set work directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for logs if it doesn't exist
RUN mkdir -p logs

# Expose the API port (Railway uses PORT env var)
EXPOSE 5050

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5050}/api/stats || exit 1

# Run the server
CMD ["python", "api_server.py"]
