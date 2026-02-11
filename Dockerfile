# Dockerfile for GodBotTrade Backend
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

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for logs and configs if they don't exist
RUN mkdir -p logs

# Expose the API port
EXPOSE 5050

# Run the server
CMD ["python", "api_server.py"]
