FROM python:3.11-slim

WORKDIR /app

# Instal dependensi sistem yang dibutuhkan oleh Prisma
RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code ke dalam kontainer
COPY . .

# Generate Prisma ORM client
RUN prisma generate

# Expose port aplikasi
EXPOSE 8000

# Jalankan aplikasi FastAPI via uvicorn
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
