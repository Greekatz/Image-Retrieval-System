import os
import faiss
import torch
import numpy as np
from glob import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Embedding:
    def __init__(self, image_folder="E:/Github/PreThesis/dataset/flickr30k_images",
                 index_path="image_embeddings.faiss", metadata_path="image_files.txt", chunk_size=256):
        """
        Initializes the embedding class for FAISS indexing using Hugging Face's CLIP model.

        :param image_folder: Path to Flickr30k image folder.
        :param index_path: File path to save FAISS index.
        :param metadata_path: File path to save image metadata.
        :param chunk_size: Number of images processed per batch (default: 256).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model from transformers
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.image_folder = image_folder
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_size = chunk_size
        self.dim = 512  # CLIP ViT-B/32 produces 512-dimensional embeddings
        self.index = faiss.IndexFlatIP(self.dim)  # Inner Product similarity
        self.index = faiss.IndexIDMap(self.index)  # Allows ID-based mapping

    def process_chunk(self, chunk):
        """
        Processes a batch of images and extracts embeddings using CLIP.

        :param chunk: List of image file paths.
        :return: NumPy array of CLIP embeddings.
        """
        images = [Image.open(img_path).convert("RGB") for img_path in chunk]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)  # Extract CLIP image features
        embeddings = embeddings.cpu().numpy()  # Convert to NumPy array
        return embeddings

    def build_index(self):
        """
        Reads images, extracts embeddings, and builds a FAISS index.
        """
        image_files = glob(os.path.join(self.image_folder, "*.jpg"))  # Adjust for different formats
        image_embeddings = []
        image_ids = []

        for i in range(0, len(image_files), self.chunk_size):
            print(f"Processing chunk {i}/{len(image_files)}...")
            chunk = image_files[i : i + self.chunk_size]
            embeddings = self.process_chunk(chunk)
            image_embeddings.append(embeddings)
            image_ids.extend(range(len(image_ids), len(image_ids) + len(embeddings)))

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

# Run embedding process when executing this file
if __name__ == "__main__":
    embedding = Embedding(image_folder="E:/Github/PreThesis/dataset/flickr30k_images")
    embedding.build_index()  # Generates and saves FAISS index
