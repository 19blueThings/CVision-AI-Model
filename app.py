from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tensorflow as tf
import pickle
import os
import io
import PyPDF2
from dotenv import load_dotenv
from google import genai

# ==========================================
# 0. LOAD ENVIRONMENT VARIABLES (.env)
# ==========================================
# Ini akan membaca file .env di folder yang sama
load_dotenv() 

# ==========================================
# 1. INISIALISASI APLIKASI & GEMINI API
# ==========================================
app = FastAPI(
    title="CVision Classifier API",
    description="API untuk Klasifikasi CV (Teks & PDF) dan Ekstraksi Saran AI"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan!")

# Inisialisasi Client baru
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# 2. REGISTRASI CUSTOM LAYER TENSORFLOW
# ==========================================
# Wajib agar tf.keras.models.load_model bisa mengenali arsitektur buatan kita
@tf.keras.utils.register_keras_serializable()
class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        super(CustomAttention, self).build(input_shape)

    def call(self, inputs):
        score = tf.matmul(inputs, self.W)
        weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * weights
        return tf.reduce_sum(context_vector, axis=1)

    # WAJIB UNTUK SERIALIZATION
    def get_config(self):
        config = super(CustomAttention, self).get_config()
        return config

# ==========================================
# 3. LOADING MODEL & METADATA (Saat Server Start)
# ==========================================
print("Memuat model Deep Learning dan konfigurasi...")
try:
    # Load Model Utama
    model = tf.keras.models.load_model(
        'CVision_Career_Classifier.keras', 
        custom_objects={'CustomAttention': CustomAttention}
    )
    
    # Load Vocabulary
    with open('vectorizer_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    # Load Nama Kategori
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
        
    # Re-build Vectorizer (Mengubah teks -> angka untuk inference)
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=10000, 
        output_mode='int', 
        output_sequence_length=300
    )
    vectorizer.set_vocabulary(vocab)
    
    print("✅ Sistem siap! Model berhasil dimuat.")
except Exception as e:
    print(f"❌ Error kritis saat memuat model: {e}")

# ==========================================
# 4. FUNGSI PEMBANTU (Membaca PDF & Core Logic)
# ==========================================
def extract_text_from_pdf(pdf_bytes):
    """Mengekstrak teks dari format PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in range(len(pdf_reader.pages)):
            extracted = pdf_reader.pages[page].extract_text()
            if extracted:
                text += extracted + " "
        return text.strip()
    except Exception as e:
        raise ValueError(f"Gagal membaca PDF: {str(e)}")

async def process_prediction(cv_text: str):
    """Fungsi inti: Melakukan prediksi dan memanggil Gemini."""
    if not cv_text:
         raise HTTPException(status_code=400, detail="Teks CV kosong.")

    try:
        # A. PREDIKSI KATEGORI (TENSORFLOW)
        vectorized_text = vectorizer([cv_text])
        prediction_scores = model(vectorized_text, training=False)
        
        predicted_index = tf.argmax(prediction_scores, axis=1).numpy()[0]
        confidence_score = tf.reduce_max(prediction_scores, axis=1).numpy()[0]
        predicted_category = class_names[predicted_index]
        
        # B. MEMINTA SARAN DARI GEMINI AI (Poin Plus)
        prompt_gemini = (
            f"Saya melamar pekerjaan di industri '{predicted_category}'. "
            f"Berikut potongan awal CV saya: '{cv_text[:400]}...' "
            f"Sebagai HRD, berikan 2 poin saran singkat dalam bahasa Indonesia "
            f"untuk memperkuat CV saya di industri tersebut."
        )
        
        # --- KODE BARU UNTUK MEMANGGIL GEMINI ---
        try:
            gemini_response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_gemini
            )
            ai_feedback = gemini_response.text
        except Exception as gemini_err:
            ai_feedback = "Sistem gagal menghubungi AI untuk memberikan saran saat ini."
            print(f"Error Gemini API: {gemini_err}")

        # C. OUTPUT JSON KE FRONTEND
        return {
            "prediksi_kategori": predicted_category,
            "confidence_score": float(confidence_score),
            "saran_pengembangan_ai": ai_feedback,
            "ekstrak_teks_awal": cv_text[:100] + "..." # Cuplikan teks agar Frontend tahu PDF berhasil dibaca
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server AI: {str(e)}")

# ==========================================
# 5. ENDPOINTS (API ROUTES)
# ==========================================

# Endpoint A: Jika Input Berupa Teks Manual
class CVRequest(BaseModel):
    teks_cv: str

@app.post("/predict_text", tags=["Prediksi Teks"])
async def predict_cv_text(request: CVRequest):
    """Prediksi CV dengan mengirimkan teks biasa."""
    return await process_prediction(request.teks_cv.strip())

# Endpoint B: Jika Input Berupa File PDF
@app.post("/predict_pdf", tags=["Prediksi File"])
async def predict_cv_pdf(file: UploadFile = File(...)):
    """Prediksi CV dengan meng-upload file format .pdf"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya menerima file berformat PDF.")
    
    try:
        pdf_content = await file.read()
        cv_text = extract_text_from_pdf(pdf_content)
        
        if not cv_text:
            raise HTTPException(status_code=400, detail="PDF kosong atau teks tidak bisa dibaca (mungkin hasil scan/gambar).")
            
        return await process_prediction(cv_text)
        
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Gagal memproses file PDF: {str(e)}")