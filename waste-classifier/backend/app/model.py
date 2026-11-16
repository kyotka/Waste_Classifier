import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

class WasteClassifier:
    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
       
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))
        
       
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        return {
            "predicted_class": CLASSES[predicted_idx.item()],
            "confidence": round(confidence.item() * 100, 2),
            "all_scores": {
                CLASSES[i]: round(probabilities[i].item() * 100, 2)
                for i in range(len(CLASSES))
            }
        }