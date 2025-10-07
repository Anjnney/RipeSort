# RipeSort

**RipeSort** is an automated, deep learning–powered fruit classification and ripeness detection system for efficient, real-time quality control and sorting. Built with PyTorch and transfer learning using MobileNetV2, RipeSort identifies both the type of fruit and its ripeness level from images or live camera input. The system is designed to modernize agricultural workflows and minimize human error in fruit grading.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Multi-Fruit Classification:** Recognizes multiple fruit types (e.g., Banana, Orange, Watermelon, etc.).
- **Ripeness Detection:** Classifies fruits as *Overripe*, *Ripe*, or *Unripe*.
- **Deep Learning Backbone:** Utilizes MobileNetV2 pretrained on ImageNet and fine-tuned for fruit/ripeness categories.
- **Transfer Learning:** Easily adaptable to new fruits or ripeness criteria by providing your own images.
- **Training & Validation Pipeline:** Includes robust training with live accuracy/loss reporting.
- **Real-Time Inference:** Supports batch image prediction and live camera input for immediate results.
- **Lightweight & Deployable:** Designed for use on CPUs and GPUs; suitable for smart agriculture and packing lines.

---

## How It Works

- RipeSort uses a transfer learning approach with MobileNetV2.
- Images are resized, normalized, and fed into the network.
- The classifier predicts both fruit type and ripeness stage based on learned patterns.
- Both batch and real-time (webcam) inference scripts are available.

---

## Installation

**Clone the repository:**

git clone https://github.com/Anjnney/RipeSort.git
cd RipeSort

**Create and activate a Python virtual environment (recommended):**

python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows


**Install dependencies:**
pip install -r requirements.txt


*(If requirements.txt is missing, see the list under "Requirements" below.)*

---

## Requirements

- Python 3.7+
- torch, torchvision (PyTorch)
- tqdm
- Pillow (PIL)
- opencv-python (only for real-time inference)
- CUDA (optional, for GPU support)

**You can install them manually:**
pip install torch torchvision tqdm Pillow opencv-python


---

## Usage

### 1. Data Organization

Place your images in the following folder structure:

RipeSort/
├── Train/
│ ├── Ripe/
│ ├── Unripe/
│ └── Overripe/
├── Test/
│ ├── Ripe/
│ ├── Unripe/
│ └── Overripe/

- Each class folder should contain images for that class.
- Add more fruit types (e.g., Banana, Orange, etc.) as additional subfolders.

### 2. Model Training

**Run the training script to fine-tune MobileNetV2:**

python main.py


- Adjust `NUM_EPOCHS`, `LR`, and `BATCH_SIZE` in `main.py` as desired.
- The trained model weights are saved to `mobilenetv2_fruit.pth`.

### 3. Model Inference

**Batch Inference:**

python inference.py


Use this for evaluating the model on test images.

**Real-time (Webcam) Inference:**

python realtime_inference.py


Classifies fruit and ripeness using your webcam stream.

- For single image prediction, use the included `predict_image()` helper function and provide an image path.

---

## Project Structure

.
├── .venv/ # Virtual environment (not included in repo)
├── Train/ # Training images (not in repo)
├── Test/ # Test images (not in repo)
├── main.py # Training and validation script
├── inference.py # Script for batch inference on test set
├── realtime_inference.py # Real-time webcam inference
├── mobilenetv2_fruit.pth # Trained weights (not included; see README)
├── .gitignore
├── LICENSE
└── README.md


---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo and create a branch (`git checkout -b feature/fooBar`)
2. Commit your changes (`git commit -am 'Add some feature'`)
3. Push to the branch (`git push origin feature/fooBar`)
4. Create a new Pull Request

Feel free to discuss ideas or request support in the issues section.

---

## License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## Contact

Project by **Anjnney**  
Linkedin - www.linkedin.com/in/anjnney-salvi-034a75288
Inspired by cutting-edge research in deep learning and computer vision for agriculture.

Ready for use in smart agriculture, R&D, and beyond.  
For questions or collaboration opportunities, open an issue or contact the project maintainer.

