# ==========================================
# Stage 1: Build Frontend (Vite + React)
# ==========================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Backend & Static Serving
# ==========================================
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies needed for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application and ML assets
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY models/ ./models/
COPY vocabulary/ ./vocabulary/

# Copy compiled frontend from Stage 1 into /app/frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose default port (Render automatically maps $PORT at runtime)
EXPOSE 8000

# Run uvicorn dynamically binding to Render's $PORT or default 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
