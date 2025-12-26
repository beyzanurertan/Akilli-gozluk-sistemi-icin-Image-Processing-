import cv2
import torch
import threading
import speech_recognition as sr
import os
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame

# model ayarı
device = "cpu"
model_id = "Salesforce/blip-image-captioning-base"

print(f"Sistem Hazırlanıyor... Model: {model_id} (CPU)")
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)
model.to(device)

# değişkenler
komut_kuyrugu = None
dinleme_aktif = True

# seslendirme
def seslendir(metin):
    if not metin or len(metin.strip()) == 0: return
    
    dosya = "yanit.mp3"
    try:
        # Google üzerinden Türkçe ses dosyası oluşturma
        tts = gTTS(text=metin, lang='tr')
        tts.save(dosya)
        
        # Pygame ile sesi çalma
        pygame.mixer.init()
        pygame.mixer.music.load(dosya)
        pygame.mixer.music.play()
        
        # Ses bitene kadar bekle
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        pygame.mixer.quit()
        
        # Dosyayı temizle
        time.sleep(0.1)
        if os.path.exists(dosya):
            os.remove(dosya)
            
    except Exception as e:
        print(f"Seslendirme Hatası: {e}")

# mikrofonu dinleme
def mikrofonu_dinle():
    global komut_kuyrugu, dinleme_aktif
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Mikrofon ayarlanıyor...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("\n>>'Önümde ne var' diyebilirsiniz.")
        
        while dinleme_aktif:
            try:
                audio = r.listen(source, timeout=3, phrase_time_limit=4)
                metin = r.recognize_google(audio, language='tr-TR').lower()
                print(f"Duyulan: {metin}")
                
                if "ne var" in metin or "görüyorsun" in metin:
                    komut_kuyrugu = "analiz"
                elif "teşekkürler" in metin or "kapat" in metin:
                    komut_kuyrugu = "cikis"
                    break
            except:
                continue

# Mikrofonu arka planda başlat
thread = threading.Thread(target=mikrofonu_dinle, daemon=True)
thread.start()

# kamera
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret: break

    # Görsel bilgilendirme
    durum = "DINLIYOR..."
    renk = (0, 255, 0)
    if komut_kuyrugu == "analiz":
        durum = "ANALIZ EDILIYOR..."
        renk = (0, 0, 255)

    cv2.putText(frame, f"Durum: {durum}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
    cv2.imshow("Goren AI (CPU - Stabil)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        dinleme_aktif = False
        break

    # komutu işleme
    if komut_kuyrugu == "analiz":
        try:
            print("--- ANALİZ BAŞLADI ---")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            start_time = time.time()
            
            # BLIP ile CPU üzerinden standart analiz
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=30)
            
            english_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            print(f"1. İngilizce: {english_caption}")

            turkish_caption = GoogleTranslator(source='auto', target='tr').translate(english_caption)
            
            # Metin temizleme 
            temiz_cevap = turkish_caption.lower().replace("bir fotoğrafı", "").replace("fotoğrafı", "").strip()
            print(f"2. Türkçe: {temiz_cevap}")
            
            # Sesli cevap verme
            seslendir(temiz_cevap)
            
        except Exception as e:
            print(f"Hata: {e}")
        
        komut_kuyrugu = None
        print(f"--- Analiz tamamlandı. ({round(time.time() - start_time, 2)} sn). ---")

    elif komut_kuyrugu == "cikis":
        seslendir("Görüşmek üzere, hoşçakalın.")
        break

cap.release()
cv2.destroyAllWindows()