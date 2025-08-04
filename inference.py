import torch
import torch.nn as nn
import cv2
import numpy as np
from models import AdvancedCNN  # Đảm bảo bạn có file models.py và class AdvancedCNN bên trong

def inference():
    image_path = "data/animals/test/cat/1.jpeg"
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AdvancedCNN()
    model.load_state_dict(torch.load("model/best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(224, 224))

    # Normalize image
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (HWC) → (CHW)
    image = torch.from_numpy(image).to(device)[None, :, :, :]  # Add batch dimension

    with torch.no_grad():
        logits = model(image)[0]  # Output shape: [num_classes]
        probs = torch.softmax(logits, dim=0)
        predicted_index = torch.argmax(probs)
        predicted_class = categories[predicted_index]

        print(f"\n✅ Predicted class: {predicted_class}")
        print("🔍 Class probabilities:")
        for idx, prob in enumerate(probs):
            print(f"  {categories[idx]:<10}: {prob.item():.4f}")

        # Display image with prediction
        original_image = cv2.imread(image_path)
        label = f"{predicted_class} ({probs[predicted_index].item() * 100:.2f}%)"
        cv2.putText(original_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Prediction", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()
