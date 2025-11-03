FROM python:3.13-slim
WORKDIR /app

# Copy sources into the image (build context is the web/ folder)
# The backend exposes `main.py` at web/backend/main.py, so copy the whole backend
# directory into /app. Install requirements from /app/requirements.txt.
COPY backend /app
COPY frontend /app/frontend

# Install curl for the docker-compose healthcheck and keep image small
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

# Run the FastAPI app with uvicorn so the container actually starts a server
# The module path is `main:app` because main.py is at /app/main.py inside the image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
