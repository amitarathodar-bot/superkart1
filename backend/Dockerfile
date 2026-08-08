# Use a highly optimized, clean Python slim base image
FROM python:3.10-slim

# Establish the workspace directory inside the container
WORKDIR /app

# Copy the dependencies configuration file first to leverage Docker layer caching
COPY requirements.txt /app/

# Install the necessary ML libraries and server packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your API application code and your existing model file into the image workspace
COPY app.py superkart_model.joblib /app/

# Inform the system that this container exposes network communication on port 5000
EXPOSE 7860

# Spin up the Flask API service
CMD ["python", "app.py"]
