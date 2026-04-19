# Use a slim version of Python
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 🚨 THE FIX: Install the tiny, CPU-only version of PyTorch first!
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Now install the rest of your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose the port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]