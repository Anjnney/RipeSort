import torch
from torchvision import models, transforms
import cv2
from PIL import Image

# --- Model and Classes Setup ---
class_names = ['Overripe', 'Ripe', 'Unripe']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load('mobilenetv2_fruit.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_frame(frame, threshold=0.5):
    # Convert OpenCV image (BGR) to PIL (RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(prob).item())
        confidence = float(prob[idx])
        if confidence < threshold:
            label = "Not a fruit or uncertain"
        else:
            label = class_names[idx]
    return label, confidence

# --- Real-Time Webcam Loop ---
cap = cv2.VideoCapture(0) # 0 is default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Predict on the center square crop for stability
    h, w, _ = frame.shape
    min_side = min(h, w)
    y1 = (h - min_side) // 2
    x1 = (w - min_side) // 2
    crop = frame[y1:y1+min_side, x1:x1+min_side]

    label, confidence = predict_frame(crop, threshold=0.5)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Fruit Ripeness Prediction', crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
