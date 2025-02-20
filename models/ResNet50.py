import torch 
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ResNet50:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """ Preprocess Image."""
        image = Image.open(image).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image
    
    def feature_extractor(self, image):
        image = self.preprocess_image(image)
        with torch.no_grad():
            image_features = self.model(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features