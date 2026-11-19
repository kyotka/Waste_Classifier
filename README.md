# Waste Material Classifier

A web application that classifies waste items into recyclable and non-recyclable categories using a fine-tuned ResNet-50 model. The app includes an object detection pre-screener (SSDLite MobileNetV3) that checks the frame first to filter out non-waste items—like people, pets, or electronics—before running the classifier.

## Supported Classes

- **Cardboard** (boxes, packaging)
- **Glass** (bottles, jars)
- **Metal** (soda cans, food tins)
- **Paper** (clean sheets, newspapers)
- **Plastic** (bottles, containers)
- **Trash** (general non-recyclable waste)

## Model Info

- **Architecture:** ResNet-50 fine-tuned on the Garbage Classification dataset
- **Pre-screener:** SSDLite320 MobileNetV3 (filters out non-waste inputs)
- **Input size:** 224x224 RGB
- **Validation accuracy:** 96.84%
- **Framework:** PyTorch & Torchvision

## How to Run

### 1. Set up virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the server

```bash
python main.py
```

Open `http://localhost:8000` in your browser. You can either drag & drop an image or use your webcam to classify an item.

### Running with Docker

```bash
docker compose up --build
```
