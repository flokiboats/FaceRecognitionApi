# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
# Install PyTorch CPU wheels first
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt 
# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
