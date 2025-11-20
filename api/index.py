# api/analyze.py
import os
import re
import json
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # إضافة CORS
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from mangum import Mangum  # Vercel adapter

app = FastAPI()

# إضافة CORS middleware عشان POST requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # أو حدد n8n domain لو عايز أمان أكتر
    allow_credentials=True,
    allow_methods=["*"],  # يسمح POST, GET, etc.
    allow_headers=["*"],
)

# DeepSeek API
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# تحميل النموذج
print("جاري تحميل نموذج الـ embeddings...")
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"}, {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"}, {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"}, {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"}, {"key": "dizziness", "text": "دوخة"},
    {"key": "voice_changes", "text": "تغيرات في جودة الصوت"}, {"key": "hoarseness", "text": "بحة الصوت"},
    {"key": "taste_changes", "text": "تغير الطعم"}, {"key": "low_appetite", "text": "انخفاض الشهية"},
    {"key": "vomiting", "text": "تقيؤ"}, {"key": "heartburn", "text": "حرقة صدر"},
    {"key": "gas", "text": "الغازات"}, {"key": "bloating", "text": "الانتفاخ"},
    {"key": "hiccups", "text": "زغطة"}, {"key": "constipation", "text": "امساك"},
    {"key": "diarrhea", "text": "اسهال"}, {"key": "fecal_incontinence", "text": "سلس برازي"},
    {"key": "breath_shortness", "text": "ضيق تنفس"},
]

symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_symptoms(user_text, threshold=0.18):
    detected = set()
    parts = re.split(r"[،,.\s!؟؛]+", user_text.lower())
    for part in parts:
        part = part.strip()
        if len(part) < 3: continue
        user_emb = model.encode([part])[0]
        similarities = [cosine_sim(user_emb, emb) for emb in symptom_embeddings]
        for idx, sim in enumerate(similarities):
            if sim > threshold:
                detected.add(SYMPTOMS[idx]["key"])
    return list(detected)

def deepseek_chat(prompt):
    payload = {
        "model": "deepseek-v3.1",
        "messages": [
            {"role": "system", "content": "أنت طبيب أورام متخصص، جاوب بالعربي الفصحى وباختصار شديد."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.3
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"خطأ في الـ AI: {r.status_code}"
    except:
        return "تعذر الاتصال بـ DeepSeek مؤقتًا."

class InputData(BaseModel):
    symptoms: str
    patient_phone: str = ""

@app.post("/api/analyze")
async def analyze(data: InputData):
    text = data.symptoms.strip()
    if not text:
        return JSONResponse({"error": "الأعراض فارغة"}, status_code=400)

    detected = detect_symptoms(text)
    
    if not detected:
        response = "لم أتمكن من التعرف على أعراض واضحة، ممكن توضح أكتر؟"
        risk = "منخفضة"
    else:
        symptoms_ar = "، ".join([s["text"] for s in SYMPTOMS if s["key"] in detected])
        prompt = f"المريض يعاني: {text}\nالأعراض المكتشفة: {symptoms_ar}\n\nأجب بالعربي فقط:\n1. الاحتمالات الطبية\n2. درجة الخطورة: منخفضة/متوسطة/عالية/طوارئ\n3. التوصية الفورية"
        response = deepseek_chat(prompt)
        risk = "طوارئ" if any(w in response for w in ["طوارئ","فورًا","مستشفى","نزيف"]) else \
               "عالية" if any(w in response for w in ["عالية","خطير","سرطان"]) else \
               "متوسطة" if "متوسطة" in response else "منخفضة"

    return {
        "medical_response": response,
        "detected_symptoms": detected,
        "risk_level": risk,
        "patient_phone": data.patient_phone,
        "raw_input": text
    }

# Vercel entry point
handler = Mangum(app, lifespan="off")
