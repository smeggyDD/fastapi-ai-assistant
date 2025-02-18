import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("thenlper/gte-large")  # Switching to a stronger model

# Load properties JSON
with open("filtered_properties.json", "r", encoding="utf-8") as f:
    properties = json.load(f)

# Generate unique embeddings
property_texts = [prop["title"] + " " + prop["location"] for prop in properties]
property_vectors = model.encode(property_texts)  # Correct embedding

# Convert to FAISS format
property_vectors = np.array(property_vectors).astype("float32")

# Save new embeddings
np.save("property_vectors.npy", property_vectors)

# Print first 5 unique values to verify
print("Unique Embeddings Sample:", property_vectors[:5])
