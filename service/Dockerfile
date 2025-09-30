# Use a lightweight Python base image
FROM python:3.13-slim

# Set work directory
WORKDIR /app

# Copy requirements first (for caching efficiency)
COPY service/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and models into container
COPY service/service.py .
COPY models ./models

# Expose FastAPI default port
EXPOSE 7000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "7000"]