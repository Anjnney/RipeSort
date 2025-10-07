import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
from tqdm import tqdm
from PIL import Image

# ----------------------
# Config
# ----------------------
DATASET_PATH = "Train"  # root folder with class subfolders
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Dataset path: {os.path.abspath(DATASET_PATH)}")

# ----------------------
# Check and print folders
# ----------------------
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"[ERROR] DATASET_PATH does not exist: {DATASET_PATH}")

class_folders = sorted(os.listdir(DATASET_PATH))
print(f"[INFO] Found class folders ({len(class_folders)}): {class_folders}")

# ----------------------
# Define transform
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MobileNet_V2_Weights.DEFAULT.transforms().mean,
                         MobileNet_V2_Weights.DEFAULT.transforms().std)
])

print("[INFO] Building ImageFolder dataset...")
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
print(f"[INFO] Classes (order used by model): {dataset.classes}")
NUM_CLASSES = len(dataset.classes)
print(f"[INFO] NUM_CLASSES set to: {NUM_CLASSES}")

# ----------------------
# Train-val split
# ----------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(f"[INFO] Train/Val split: {train_size}/{val_size}")
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Peek batch (optional)
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"[INFO] One batch images shape: {images.shape}")
print(f"[INFO] One batch labels: {labels[:16].tolist()} ...")

# ----------------------
# Model
# ----------------------
print("[INFO] Loading MobileNetV2 with ImageNet weights...")
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------
# Training loop
# ----------------------
print("[INFO] Starting training...")
for epoch in range(NUM_EPOCHS):
    start_t = time.time()
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for imgs, labs in pbar:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outs.argmax(1)
        running_correct += (preds == labs).sum().item()
        running_total += labs.size(0)

        pbar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "acc": f"{(running_correct / running_total) * 100:.2f}%"
        })

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    # Validation
    model.eval()
    val_correct, val_total, val_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ", leave=False)
        for imgs, labs in pbar_val:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            outs = model(imgs)
            loss = criterion(outs, labs)
            val_loss_sum += loss.item() * imgs.size(0)
            preds = outs.argmax(1)
            val_correct += (preds == labs).sum().item()
            val_total += labs.size(0)

    val_loss = val_loss_sum / max(1, val_total)
    val_acc = val_correct / max(1, val_total)

    dur = time.time() - start_t
    print(f"[EPOCH {epoch+1}/{NUM_EPOCHS}] "
          f"train_loss={epoch_loss:.4f} train_acc={epoch_acc*100:.2f}% | "
          f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
          f"time={dur:.1f}s")

# ----------------------
# Save model
# ----------------------
save_path = 'mobilenetv2_fruit.pth'
torch.save(model.state_dict(), save_path)
print(f"[INFO] Model saved to: {os.path.abspath(save_path)}")

# ----------------------
# Inference helper and file listing
# ----------------------
def predict_image(path, model, class_names):
    print(f"[INFO] Predicting: {path}")
    model.eval()
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        pred_idx = int(torch.argmax(prob).item())
        print(f"[RESULT] class='{class_names[pred_idx]}' prob={float(prob[pred_idx]):.4f}")
    return class_names[pred_idx], prob.cpu().numpy()

# List available test images for each class
for folder in dataset.classes:
    dir_path = os.path.join('Test', folder)
    print(f"Files in {dir_path}:")
    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            print("  ", f)

# Pick any image from above output and use it here
# Example:
predict_image('Test/Ripe/apple_ripe_004.jpg', model, dataset.classes)

print("[INFO] Ready for inference. Replace <your_image_name_here> above with real file name from output.")
