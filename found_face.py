import torch 
import cv2
import os
from torchvision import models
from torchvision import transforms
import mediapipe as mp
from PIL import Image
import torch.nn as nn

def draw(frame_copy,bboxC):
    h, w, c = frame.shape
    x_min = int(bboxC.xmin * w)
    y_min = int(bboxC.ymin * h)
    box_width = int(bboxC.width * w)
    box_height = int(bboxC.height * h)
    
    # Yüz bölgesini kırp
    face_region = frame_copy[y_min:y_min + box_height, x_min:x_min + box_width]
    
    if face_region.size != 0:
        
        cls_name, score = cnn(face_region)  # CNN modelini çalıştır
    
        cv2.putText(frame, f"{cls_name}: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)
        return frame

def göster(frame):
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q' tuşuna basarak çıkış yapabilirsiniz
        return True
    return False

def cnn(frame_copy):
    frame_copy = cv2.resize(frame_copy, (224, 224))  # Boyutlandırma
    frame_copy = Image.fromarray(frame_copy)  # OpenCV'den PIL'e dönüştürme
    frame_copy = transform(frame_copy).unsqueeze(0).to(device)  # Tensor haline getiriyoruz
    
    class_names = ['person', 'ronaldo']  # Sınıflar
    model.eval()  # Modeli değerlendirme moduna al
    
    with torch.no_grad():
        output = model(frame_copy)  # Modelden çıktı al
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)  # En yüksek olasılıklı sınıfı al
        cls_name = class_names[predicted_class]
        
        skor = probabilities[0][predicted_class.item()].item()
        print(f"{cls_name} | {skor*100:.0f}")  # Sınıf ve güven skoru ekrana yazdır
        return cls_name, skor  # Model çıktısını döndür

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VERİ ÖN İŞLEME
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MODEL YÜKLEME
model_path = "person_ronaldo_model.pth"
model = torch.load(model_path, map_location=device)

# Kamera ayarları
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Kamera açılamadı!")

# Mediapipe ile yüz tespiti
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        break

    frame = cv2.flip(frame, 1)  
    results = mp_face_detection.process(frame)  # Yüz tespiti yap

    frame_copy = frame.copy() 
    
    # Eğer yüz tespit edilirse
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            draw(frame_copy,bboxC)
            

    if göster(frame):
        break

cap.release()
cv2.destroyAllWindows()
