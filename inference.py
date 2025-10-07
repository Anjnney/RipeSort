import torch
from torchvision import models, transforms
from PIL import Image
import os #for batch testing

# Load class names as used during training
class_names = ['Overripe', 'Ripe', 'Unripe']  # Use the actual class order you had

# Device and transforms -- must match training!
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Build MobileNetV2 and load weights (NUM_CLASSES = 3)
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load('mobilenetv2_fruit.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_image(path):
    print(f"Predicting: {path}")
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(prob).item())
        print(f"Predicted class: '{class_names[idx]}', probability: {float(prob[idx]):.4f}")
    return class_names[idx], prob.cpu().numpy()

# Example: test one image
#predict_image(r"Test/Ripe/apple_ripe_004.jpg")

test_root = "Test"
total = 0
class_correct = {c: 0 for c in class_names}
class_total = {c: 0 for c in class_names}

for true_class in class_names:
    folder = os.path.join(test_root, true_class)
    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        continue
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        pred, _ = predict_image(img_path)
        class_total[true_class] += 1
        total += 1
        if pred == true_class:
            class_correct[true_class] += 1

print("\n[Batch Test Results]")
for c in class_names:
    c_total = class_total[c]
    c_corr = class_correct[c]
    acc = 100 * c_corr / c_total if c_total else 0
    print(f"{c}: {c_corr}/{c_total} correct ({acc:.2f}%)")
overall_acc = 100 * sum(class_correct.values()) / total if total else 0
print(f"Overall test accuracy: {overall_acc:.2f}%")
