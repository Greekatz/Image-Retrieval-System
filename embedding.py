import os
import faiss
import torch
import numpy as np
from glob import glob
from PIL import Image
from models.CLIP import CLIP
from models.VGG19 import VGG19 
from models.ResNet50 import ResNet50

class Embedding:
    def __init__(self, image_folder="E:/Github/PreThesis/dataset/flickr30k_images",
                 index_path="image_embeddings.faiss", metadata_path="image_files.txt", chunk_size=256, model_type="clip"):
        """
        Initializes the embedding class for FAISS indexing using Hugging Face's CLIP model.

        :param image_folder: Path to Flickr30k image folder.
        :param index_path: File path to save FAISS index.
        :param metadata_path: File path to save image metadata.
        :param chunk_size: Number of images processed per batch (default: 256).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_folder = image_folder
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_size = chunk_size
        self.model_type = model_type.lower()
        
        # Load the selected model
        self.model, self.dim = self.load_model()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dim)  # Inner Product similarity
        self.index = faiss.IndexIDMap(self.index)  # Allows ID-based mapping
    
    
    def load_model(self):
        """
        Loads the model for feature extraction.
        """
        if self.model_type == "clip":
            model = CLIP()
            dim = 512
        elif self.model_type == "vgg19":
            model =  VGG19()
            dim = 512
        elif self.model_type == "resnet50":
            model = ResNet50()
            dim = 2048
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        return model, dim

    def process_chunk(self, chunk):
        """
        Processes a batch of images and extracts embeddings using the selected model.
        """
        images = [Image.open(img_path).convert("RGB") for img_path in chunk]
        embeddings = np.array([self.model.feature_extractor(img) for img in images], dtype="float32")
        embeddings = embeddings.reshape(len(images), -1)
        return embeddings

    def build_index(self):
        """
        Reads images, extracts embeddings, and builds a FAISS index.
        """
        image_files = glob(os.path.join(self.image_folder, "*.jpg"))  
        image_embeddings = []
        image_ids = []

        start_id = 0
        for i in range(0, len(image_files), self.chunk_size):
            print(f"Processing chunk {i}/{len(image_files)}")
            chunk = image_files[i : i + self.chunk_size]
            embeddings = self.process_chunk(chunk)
            print(f"Chunk {i}: Embedding shape: {embeddings.shape}")
            image_embeddings.append(embeddings)
            image_ids.extend(range(start_id, start_id + len(embeddings)))
            start_id += len(embeddings)

        image_embeddings = np.vstack(image_embeddings)
        vectors = np.array(image_embeddings, dtype="float32")
        ids = np.array(image_ids)

        self.index.add_with_ids(vectors, ids)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved at {self.index_path}")

        # Save image file paths
        with open(self.metadata_path, "w") as f:
            for image_file in image_files:
                f.write(image_file + "\n")
        print(f"Image metadata saved at {self.metadata_path}")

    def load_index(self):
        """
        Loads the FAISS index if it exists.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("FAISS index loaded successfully.")

if __name__ == "__main__":
    model_types = ["vgg19","clip"]
    for model in model_types:
        print(f"Building index with {model}...")
        embedding = Embedding(image_folder="E:/Github/PreThesis/dataset/flickr30k_images", model_type=model)
        embedding.build_index()
