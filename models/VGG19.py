import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG19_Weights
from PIL import Image



class VGG19:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Load and preprocess a single image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image
    
    def feature_extractor(self, image):
        """Extract features from an image."""

        image = self.preprocess_image(image)
        with torch.no_grad():
            image_features = self.model(image)

           
            image_features = torch.nn.functional.adaptive_avg_pool2d(image_features, (1, 1))
            image_features = image_features.view(image_features.size(0), -1)  
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features  
