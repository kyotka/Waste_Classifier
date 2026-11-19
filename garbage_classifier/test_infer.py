from PIL import Image, ImageDraw
import io
from model import GarbageClassifier

def test():
    classifier = GarbageClassifier("best_resnet50_garbage.pth")
    img = Image.new("RGB", (224, 224), color=(180, 130, 80))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 180, 180], fill=(120, 80, 40))
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    res = classifier.predict(image_bytes)
    print("Inference successful:", res["predicted_class"], f"({res['confidence']}%)")

if __name__ == "__main__":
    test()
