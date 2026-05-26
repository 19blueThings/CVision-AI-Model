# 🚀 CVision Career Classifier API

CVision API adalah layanan *backend* berbasis *Machine Learning* yang bertugas membaca *Curriculum Vitae* (CV) pelamar, memprediksi industri kategori bidang yang paling relevan, dan memberikan *feedback* profesional layaknya seorang HRD.

Proyek ini menggabungkan dua lapis kecerdasan buatan:
1. **Predictive AI (NLP):** Menggunakan arsitektur Deep Learning **Bidirectional LSTM + Custom Attention Mechanism** untuk mengklasifikasikan CV ke dalam berbagai kategori industri.
2. **Generative AI:** Mengintegrasikan **Google Gemini 2.0 Flash** untuk membaca konteks CV dan memberikan saran perbaikan yang konstruktif.

---

## 🛠️ Tech Stack
- **Framework API:** FastAPI & Uvicorn
- **Machine Learning Core:** TensorFlow & Keras
- **Generative AI:** Google GenAI SDK (Gemini)
- **Utility:** PyPDF2 (PDF Extractor), Python-Dotenv

---

## 💻 Cara Menjalankan di Localhost (Untuk Tim Fullstack)

Ikuti langkah-langkah di bawah ini untuk menyalakan API di laptop masing-masing:

### 1. Clone Repository/Download ZIP
Pastikan Anda sudah mengunduh *repository* ini dan masuk ke dalam foldernya melalui terminal CMD:
```bash
git clone https://github.com/19blueThings/CVision-AI-Model.git
cd CVision-AI-Model
code . 

### 2. Setup Virtual Environment
Sangat disarankan menggunakan virtual environment agar library TensorFlow tidak bentrok dengan aplikasi Python lain di laptop Anda.

# Membuat virtual environment bernama "env"
python -m venv env

# Aktivasi (Pengguna Windows):
env\Scripts\activate

# Aktivasi (Pengguna Mac/Linux):
source env/bin/activate

### 3. Install Dependencies
Instal semua library yang dibutuhkan melalui requirements.txt:
pip install -r requirements.txt

### 4. Setup Environment Variables (.env)
Aplikasi ini membutuhkan API Key dari Google Gemini.

Buat file baru bernama .env di dalam folder root.
Buka file .env
Ganti nilainya dengan API Key yang valid:

GEMINI_API_KEY=masukkan_api_key_disini

### 5. Jalankan Server
Nyalakan server FastAPI menggunakan Uvicorn:
uvicorn app:app --reload

✅ Jika berhasil, server akan menyala di: http://127.0.0.1:8000

(Anda bisa membuka http://127.0.0.1:8000/docs di browser untuk melakukan testing API melalui antarmuka interaktif Swagger UI).

-----

📡 API Documentation & Endpoint
Untuk kebutuhan integrasi dengan Frontend, silakan gunakan endpoint utama di bawah ini:

### 1. Upload & Klasifikasi CV PDF
Menerima file PDF, mengekstrak teksnya, melakukan prediksi kategori industri, dan mengembalikan saran HRD dari Gemini.

URL: /predict_pdf
Method: POST
Content-Type: multipart/form-data

Request Body
Key: file
Type: file
Description: File CV pelamar dalam format .pdf (Wajib)

--Contoh Penggunaan (JavaScript / Fetch)--
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("[http://127.0.0.1:8000/predict_pdf](http://127.0.0.1:8000/predict_pdf)", {
  method: "POST",
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

✅ Success Response (200 OK)
Perhatikan keys di bawah ini untuk ditampilkan pada UI/Desain Frontend:
JSON
{
  "status": "success",
  "filename": "cv_johndoe.pdf",
  "prediksi_kategori": "IT & ENGINEERING",
  "confidence_score": "95.5%",
  "saran_pengembangan_ai": "CV Anda sudah sangat kuat di bidang IT. Saran dari HRD: Tambahkan metrik pencapaian pada pengalaman kerja sebelumnya (misal: 'Meningkatkan kecepatan loading website hingga 40%'). Ini akan membuat CV Anda lebih menonjol di mata rekruter."
}

❌ Error Response (Contoh 400 Bad Request)
{
  "detail": "File harus berupa PDF."
}

Note untuk Tim Frontend: API ini sudah dikonfigurasi menggunakan CORS Middleware (allow_origins=["*"]), sehingga Anda bisa langsung menembak endpoint ini dari aplikasi React/Vue/Vanilla JS Anda di localhost tanpa terkena CORS Blocked.