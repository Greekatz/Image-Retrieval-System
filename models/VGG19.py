import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class VGG19:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image
    
    def extract_features(self, image_path):
        """Extract features from the given image path"""
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            features = self.model(image)
        return features
    