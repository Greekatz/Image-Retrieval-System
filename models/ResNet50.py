import torch 
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

class ResNet50:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """ Preprocess Image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image
    
    def feature_extractor(self, image):
        """Extract features from an image."""
        image = self.preprocess_image(image)

        # Debugging: Check input shape before passing to model
        print(f"Input shape to ResNet50: {image.shape}")  # Should be (1, 3, 224, 224)

        with torch.no_grad():
            image_features = self.model(image)

            # Ensure feature extraction is working
            print(f"Raw extracted feature shape: {image_features.shape}")  

            image_features = torch.nn.functional.adaptive_avg_pool2d(image_features, (1, 1))  # Global Pooling
            image_features = image_features.view(image_features.size(0), -1)  # Flatten
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize

        return image_features
