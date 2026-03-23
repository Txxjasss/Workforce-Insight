FROM python:3.11-slim

WORKDIR /app

# Install the CPU-only PyTorch wheel first (smaller, no CUDA) then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
