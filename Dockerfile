# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies (for opencv, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories (static/images etc.)
RUN mkdir -p static/images

# Expose port (Railway sets PORT, but good practice)
EXPOSE 8000

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Default command to run the app via gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
