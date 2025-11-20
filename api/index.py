# api/analyze.py
import os
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# DeepSeek API (Huawei Cloud) – ضع الكي في Environment Variables
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")  # هتضيفه في Vercel

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# تحميل النموذج مرة واحدة عند بدء التشغيل
print("جاري تحميل نموذج الـ embeddings...")
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"},
    {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"},
    {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"},
    {"key": "dizziness", "text": "دوخة"},
    {"key": "voice_changes", "text": "تغيرات في جودة الصوت"},
    {"key": "hoarseness", "text": "بحة الصوت"},
    {"key": "taste_changes", "text": "تغير الطعم"},
    {"key": "low_appetite", "text": "انخفاض الشهية"},
    {"key": "vomiting", "text": "تقيؤ"},
    {"key": "heartburn", "text": "حرقة صدر"},
    {"key": "gas", "text": "الغازات"},
    {"key": "bloating", "text": "الانتفاخ"},
    {"key": "hiccups", "text": "زغطة"},
    {"key": "constipation", "text": "امساك"},
    {"key": "diarrhea", "text": "اسهال"},
    {"key": "fecal_incontinence", "text": "سلس برازي"},
    {"key": "breath_shortness", "text": "ضيق تنفس"},
]

# حساب الـ embeddings مرة واحدة
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

def detect_symptoms(user_text, top_k=5, threshold=0.18):
    import re
    parts = re.split(r"[,.!؟؛،\s]+", user_text)
    detected = set()

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for part in parts:
        part = part.strip()
        if len(part) < 3: continue
        user_emb = model.encode([part])[0]
        similarities = [cosine_sim(user_emb, emb) for emb in symptom_embeddings]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        for idx in top_indices:
            if similarities[idx] > threshold:
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
        response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"خطأ في الاتصال بالـ AI: {response.status_code}"
    except:
        return "تعذر الاتصال بـ DeepSeek حاليًا."

class RequestBody(BaseModel):
    symptoms: str
    patient_phone: str = ""

@app.post("/api/analyze")
async def analyze(body: RequestBody):
    user_text = body.symptoms.strip()
    if not user_text:
        return {"error": "لا توجد أعراض"}

    # 1. كشف الأعراض
    detected_keys = detect_symptoms(user_text)

    if not detected_keys:
        ai_response = "لم أتمكن من التعرف على أي أعراض واضحة، برجاء وصفها بطريقة أوضح."
        "risk_level منخفض."
    else:
        # 2. توليد التحليل الطبي
        symptoms_arabic = "، ".join([s["text"] for s in SYMPTOMS if s["key"] in detected_keys])
        prompt = f"""
        المريض يعاني من الأعراض التالية: {user_text}
        الأعراض المكتشفة: {symptoms_arabic}

        أجب بالعربي فقط:
        1. ما الاحتمالات الطبية الأكثر شيوعًا؟
        2. درجة الخطورة: منخفضة / متوسطة / عالية / طوارئ
        3. التوصية الفورية للمريض
        """
        ai_response = deepseek_chat(prompt)

    # تحديد درجة الخطورة تلقائيًا
    risk = "طوارئ" if any(x in ai_response for x in ["طوارئ", "فورًا", "مستشفى", "نزيف", "ضيق تنفس شديد"]) \
          else "عالية" if any(x in ai_response for x in ["عالية", "خطير", "سرطان"]) \
          else "متوسطة" if "متوسطة" in ai_response else "منخفضة"

    return {
        "medical_response": ai_response,
        "detected_symptoms": detected_keys,
        "risk_level": risk,
        "patient_phone": body.patient_phone,
        "raw_input": user_text
    }

# للتجربة المحلية
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)