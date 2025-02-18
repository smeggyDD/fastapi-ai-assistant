import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load property data
with open("filtered_properties.json", "r", encoding="utf-8") as f:
    properties = json.load(f)

# Load sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert property descriptions to vectors
property_texts = [prop["title"] + " " + prop["location"] + " " + prop["category"] + " " + prop["property_type"] for prop in properties]
property_vectors = model.encode(property_texts).astype(np.float32)

# Define FAISS index (auto-detect dimensions)
embedding_dim = property_vectors.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index

# Add vectors to the FAISS index
index.add(property_vectors)

# Save FAISS index
faiss.write_index(index, "property_index.faiss")

print(f"✅ FAISS index created with {len(properties)} properties and saved as 'property_index.faiss'.")
