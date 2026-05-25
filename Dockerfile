# Menggunakan sistem operasi Linux ringan dengan Python bawaan
FROM python:3.10-slim

# Menentukan direktori kerja di dalam kontainer
WORKDIR /app

# Menyalin file daftar pustaka terlebih dahulu untuk efisiensi
COPY requirements.txt .

# Menginstal semua dependensi
# PENTING: Pastikan di dalam requirements.txt ada versi spesifik, misalnya tensorflow==2.14.0
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file model, konfigurasi, dan app.py ke dalam kontainer
COPY . .

# Perintah untuk menjalankan FastAPI saat kontainer menyala
# Menggunakan variabel $PORT dari Railway. Jika tidak ada, default ke 8080.
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"