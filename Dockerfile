FROM python:3.9-slim

WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .

# Install the dependencies from requirements.txt, and also install streamlit
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir streamlit

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
