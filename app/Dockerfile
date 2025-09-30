# Use an official lightweight Python image
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements (explicit or generate inline)
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code into container
COPY app/app.py .

# Expose Streamlit default port
EXPOSE 7001

# Streamlit runs in container on 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=7001", "--server.address=0.0.0.0"]
