# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY FastAPI/ .

# Download NLTK data required by your script
# This step is crucial because the script downloads it at runtime
# We'll do it during the build process to avoid runtime issues
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn to serve the application
# Assuming your main script is named 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]