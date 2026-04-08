FROM python:3.10-slim

WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Run app
CMD ["python", "server/app.py"]
