# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and templates
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["python", "app.py"]