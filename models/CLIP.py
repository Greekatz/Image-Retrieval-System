import torch
from transformers import CLIPProcessor, CLIPModel

class CLIP:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def image_feature_extractor(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs
    
    def text_feature_extractor(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs
    
    def similarity_score(self, image, text):
        image_features = self.image_feature_extractor(image)
        text_features = self.text_feature_extractor(text)
        with torch.no_grad():
            similarity = (image_features @ text_features.T).mean().item()
        return similarity
    