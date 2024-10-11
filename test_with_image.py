# library

import torch 
from torchvision import models
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Veri Ön işleme

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "person_ronaldo_model.pth"
# Modelin CPU'ya yüklenmesi
model = torch.load(model_path, map_location=device)

test_folder = "test/ronaldo"
out_folder = "out_folder"

if not os.path.exists(out_folder):
    os.mkdir(out_folder)



def göster(real_image, cls_name, skor,image_path):
    real_image = cv2.cvtColor(real_image,cv2.COLOR_BGR2RGB)
    cv2.putText(real_image, f"{cls_name}: {skor:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.imshow("img", real_image)
    
    base_name = os.path.basename(image_path)  # Dosya adını al
    output_path = os.path.join(out_folder, base_name)  # out_folder içine tam yol
    print(image_path)
    cv2.imwrite(output_path, real_image)
    
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):  # "q" tuşuna basılırsa çık
        cv2.destroyAllWindows()
    
    
    
def tespit(model,img,image_path):
    real_image = img.copy()
    
    image = Image.fromarray(real_image)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    class_names = ['person', 'ronaldo']
    num_classes = len(class_names) 
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        cls_name = class_names[predicted_class]
        
        skor = probabilities[0][predicted_class.item()].item()
        #print(f"{cls_name} {(skor):.2f}")
        
        return göster(real_image,cls_name,skor,image_path)

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

    for file in os.listdir(test_folder):
       image_path = os.path.join(test_folder, file)  
       image_path = image_path.replace("\\","/")
       image = cv2.imread(image_path)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       results = face_detection.process(image)
       if results.detections:
           tespit(model,image,image_path)
           
           
           
           
       
           

    
