# Use a Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY backend-requirements.txt /app/requirements.txt
# Or unravel-ai/requirements.txt if that's the correct one

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . /app

# Expose the port your application runs on (e.g., for FastAPI)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Adjust "src.api.main:app" if your main file and app instance have a different path/name
