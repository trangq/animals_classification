import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models import resnet18

def inference():
    image_path = "data/animals/test/cat/1.jpeg"
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = resnet18()
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    model.load_state_dict(torch.load("best.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(224, 224))
    image = np.transpose(image, (2, 0, 1)) / 255.0  # HWC -> CHW + normalize
    image = image.astype(np.float32)
    image = torch.from_numpy(image).to(device)[None, :, :, :]  # add batch dim

    softmax = nn.Softmax(dim=0)

    with torch.no_grad():
        logits = model(image)[0]
        probs = softmax(logits)
        predicted_index = torch.argmax(probs)
        predicted_class = categories[predicted_index]

        print(f"Predicted class: {predicted_class}")
        print("Class probabilities:")
        for idx, prob in enumerate(probs):
            print(f"{categories[idx]}: {prob.item():.4f}")

        # Show image with prediction
        original_image = cv2.imread(image_path)
        label = f"{predicted_class} ({probs[predicted_index].item() * 100:.2f}%)"
        cv2.imshow(label, original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()
