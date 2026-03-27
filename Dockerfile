# Dockerfile

FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the application
CMD [ "streamlit", "run", "your_script.py"]