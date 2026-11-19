import io
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torchvision.transforms.functional as TF
from PIL import Image

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

CLASS_INFO = {
    'cardboard': {
        'name': 'Cardboard',
        'category': 'Cardboard',
        'recyclable': True,
        'bin_color': 'Blue Bin (Paper & Cardboard)',
        'advice': 'Fully recyclable material. Flatten boxes to save bin space. Ensure dry and free of excessive adhesive tape or food contamination.'
    },
    'glass': {
        'name': 'Glass',
        'category': 'Glass',
        'recyclable': True,
        'bin_color': 'Green Bin (Glass Packaging)',
        'advice': 'Infinitely recyclable without loss of quality. Rinse jars and bottles. Remove metal lids or plastic caps before disposal.'
    },
    'metal': {
        'name': 'Metal & Aluminum',
        'category': 'Metal',
        'recyclable': True,
        'bin_color': 'Yellow Bin (Metals & Cans)',
        'advice': 'Includes drink cans and steel food tins. Rinse food residues and compress cans if possible to conserve bin volume.'
    },
    'paper': {
        'name': 'Paper',
        'category': 'Paper',
        'recyclable': True,
        'bin_color': 'Blue Bin (Clean Dry Paper)',
        'advice': 'Recyclable if clean and dry. Soiled or grease-contaminated paper belongs in general municipal waste.'
    },
    'plastic': {
        'name': 'Plastic',
        'category': 'Plastic',
        'recyclable': True,
        'bin_color': 'Yellow / Orange Bin (Plastics)',
        'advice': 'Includes recyclable bottles, jugs, and containers. Empty all liquids and crush the container before disposal.'
    },
    'trash': {
        'name': 'General Waste',
        'category': 'General Waste',
        'recyclable': False,
        'bin_color': 'Black Bin (Residual General Waste)',
        'advice': 'Mixed composite or non-recyclable refuse. Dispose of securely in municipal general waste containers.'
    },
    'unknown': {
        'name': 'Unrecognized Object',
        'category': 'Unrecognized Object',
        'recyclable': False,
        'bin_color': 'Manual Inspection / Uncategorized',
        'advice': 'Confidence score is below the verification threshold. The item does not clearly match any of the six supported waste classes.'
    }
}

NON_WASTE_LABELS = {
    'person': 'Person / Human Body',
    'cat': 'Cat / Pet Animal',
    'dog': 'Dog / Pet Animal',
    'horse': 'Horse / Animal',
    'sheep': 'Sheep / Animal',
    'cow': 'Cow / Animal',
    'elephant': 'Elephant / Animal',
    'bear': 'Bear / Animal',
    'zebra': 'Zebra / Animal',
    'giraffe': 'Giraffe / Animal',
    'bird': 'Bird / Animal',
    'car': 'Car / Motor Vehicle',
    'motorcycle': 'Motorcycle',
    'airplane': 'Aircraft / Airplane',
    'bus': 'Bus / Public Transit',
    'train': 'Train / Locomotive',
    'truck': 'Truck / Commercial Vehicle',
    'boat': 'Boat / Watercraft',
    'bicycle': 'Bicycle',
    'laptop': 'Laptop Computer',
    'cell phone': 'Mobile Phone',
    'keyboard': 'Computer Keyboard',
    'mouse': 'Computer Mouse',
    'remote': 'Remote Control',
    'tv': 'Television / Computer Monitor',
    'chair': 'Chair / Furniture',
    'couch': 'Sofa / Couch',
    'bed': 'Bed / Furniture',
    'dining table': 'Dining Table',
    'toilet': 'Toilet / Sanitary Fixture',
    'microwave': 'Microwave Oven',
    'oven': 'Kitchen Oven',
    'toaster': 'Toaster Appliance',
    'refrigerator': 'Refrigerator',
    'clock': 'Wall Clock / Timepiece'
}

WASTE_LIKE_LABELS = {
    'bottle': 'Bottle Container',
    'wine glass': 'Glassware / Wine Glass',
    'cup': 'Cup / Mug',
    'fork': 'Cutlery (Fork)',
    'knife': 'Cutlery (Knife)',
    'spoon': 'Cutlery (Spoon)',
    'bowl': 'Bowl Container',
    'book': 'Book / Printed Paper',
    'vase': 'Vase / Glass Container',
    'scissors': 'Scissors'
}

class WastePreScreenDetector:
    def __init__(self, device: torch.device):
        self.device = device
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.categories = weights.meta['categories']

    def inspect(self, image: Image.Image) -> dict:
        grayscale_arr = np.array(image.convert("L"), dtype=np.float32)
        std_val = float(np.std(grayscale_arr))
        if std_val < 14.0:
            return {
                "valid": False,
                "reason_code": "BLANK_OR_NO_CONTRAST",
                "detected_object": "None / Low Contrast",
                "message": "Frame is blank, dark, or lacks contrast. Please point the camera at a tangible waste object."
            }

        tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)[0]

        boxes = preds.get('boxes', [])
        labels = preds.get('labels', [])
        scores = preds.get('scores', [])

        detected_non_waste = []
        detected_waste_candidate = []

        for box, lbl_idx, score in zip(boxes, labels, scores):
            score_val = float(score.item())
            if score_val < 0.40:
                continue

            lbl_name = self.categories[lbl_idx.item()]
            if lbl_name in NON_WASTE_LABELS:
                detected_non_waste.append({
                    "label": lbl_name,
                    "display": NON_WASTE_LABELS[lbl_name],
                    "score": round(score_val * 100, 1),
                    "box": [round(float(v), 1) for v in box.tolist()]
                })
            elif lbl_name in WASTE_LIKE_LABELS:
                detected_waste_candidate.append({
                    "label": lbl_name,
                    "display": WASTE_LIKE_LABELS[lbl_name],
                    "score": round(score_val * 100, 1),
                    "box": [round(float(v), 1) for v in box.tolist()]
                })

        if detected_non_waste:
            primary = detected_non_waste[0]
            return {
                "valid": False,
                "reason_code": "NON_WASTE_OBJECT",
                "detected_object": primary["label"],
                "detected_display": primary["display"],
                "score": primary["score"],
                "bounding_box": primary["box"],
                "message": f"Non-waste object detected: '{primary['display']}' ({primary['score']}%). This system is designed solely for recyclable materials and waste items."
            }

        best_candidate = detected_waste_candidate[0] if detected_waste_candidate else None

        return {
            "valid": True,
            "detected_object": best_candidate["label"] if best_candidate else None,
            "detected_display": best_candidate["display"] if best_candidate else None,
            "bounding_box": best_candidate["box"] if best_candidate else None
        }

class GarbageClassifier:
    def __init__(self, checkpoint_path: str = "best_resnet50_garbage.pth", device: str = None, confidence_threshold: float = 55.0):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        self.class_names = CLASS_NAMES
        self.confidence_threshold = confidence_threshold
        self.img_size = 224
        self.best_acc = None

        self.model = self._load_model()
        self.pre_screener = WastePreScreenDetector(device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, len(self.class_names))
        )

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            if "class_names" in ckpt:
                self.class_names = ckpt["class_names"]
            if "img_size" in ckpt:
                self.img_size = ckpt["img_size"]
            if "best_acc" in ckpt:
                self.best_acc = ckpt["best_acc"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes: bytes):
        start_time = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inspection = self.pre_screener.inspect(image)
        if not inspection["valid"]:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "rejected",
                "is_valid_waste": False,
                "predicted_class": "unsuitable_input",
                "rejection_code": inspection.get("reason_code"),
                "rejection_reason": inspection.get("message"),
                "detected_object": inspection.get("detected_object"),
                "detected_display": inspection.get("detected_display"),
                "category": "Unsuitable for Waste Classification",
                "name": "Unsuitable for Waste Classification",
                "confidence": inspection.get("score", 0.0),
                "recyclable": False,
                "bin_color": "Excluded (Rejected by Object Detector)",
                "advice": inspection.get("message"),
                "all_predictions": [],
                "inference_time_ms": elapsed_ms,
                "device": str(self.device)
            }

        crop_target = image
        bbox = inspection.get("bounding_box")
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            w, h = image.size
            pad_x = (xmax - xmin) * 0.10
            pad_y = (ymax - ymin) * 0.10
            crop_box = (
                max(0, int(xmin - pad_x)),
                max(0, int(ymin - pad_y)),
                min(w, int(xmax + pad_x)),
                min(h, int(ymax + pad_y))
            )
            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                crop_target = image.crop(crop_box)

        tensor = self.transform(crop_target).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_prob, top_idx = torch.topk(probabilities, 1)
        pred_idx = top_idx.item()
        pred_class = self.class_names[pred_idx]
        confidence = float(top_prob.item() * 100)

        all_predictions = []
        for idx, class_name in enumerate(self.class_names):
            prob = float(probabilities[idx].item() * 100)
            info = CLASS_INFO.get(class_name, {})
            all_predictions.append({
                "class_name": class_name,
                "category": info.get("category", class_name),
                "confidence": round(prob, 2),
                "recyclable": info.get("recyclable", False)
            })

        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        top_info = CLASS_INFO.get(pred_class, {})

        is_unknown = confidence < self.confidence_threshold
        if is_unknown:
            display_class = "unknown"
            name = "Unrecognized Object"
            category = "Unrecognized Object"
            recyclable = False
            bin_color = "Manual Inspection / Uncategorized"
            advice = f"Confidence score ({confidence:.1f}%) is below the required verification threshold ({self.confidence_threshold:.0f}%). The item does not clearly match any of the six supported waste classes."
        else:
            display_class = pred_class
            name = top_info.get("name", pred_class)
            category = top_info.get("category", pred_class)
            recyclable = top_info.get("recyclable", False)
            bin_color = top_info.get("bin_color", "")
            advice = top_info.get("advice", "")

        return {
            "status": "classified",
            "is_valid_waste": True,
            "predicted_class": display_class,
            "raw_predicted_class": pred_class,
            "is_unknown": is_unknown,
            "detected_candidate": inspection.get("detected_display"),
            "confidence_threshold": self.confidence_threshold,
            "name": name,
            "category": category,
            "confidence": round(confidence, 2),
            "recyclable": recyclable,
            "bin_color": bin_color,
            "advice": advice,
            "all_predictions": all_predictions,
            "inference_time_ms": elapsed_ms,
            "device": str(self.device)
        }
