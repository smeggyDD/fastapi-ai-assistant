import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# Load your property data
with open("filtered_properties.json", "r", encoding="utf-8") as file:
    properties = json.load(file)

# Load a sentence transformer model (lightweight & effective)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare data for embedding
texts = []
property_details = {}

for idx, property in enumerate(properties):
    text = f"{property['title']} in {property['location']}, {property['category']} with {property['bedrooms']} beds and {property['bathrooms']} baths."
    texts.append(text)
    property_details[idx] = property  # Store details for later retrieval

# Convert texts to embeddings
# Convert texts to embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Save embeddings as property_vectors.npy (optional)
np.save("property_vectors.npy", embeddings)

# Create FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save FAISS index and property details
faiss.write_index(index, "property_index.faiss")

with open("property_metadata.json", "w", encoding="utf-8") as meta_file:
    json.dump(property_details, meta_file, indent=4)

print("✅ Property data successfully embedded and stored in FAISS!")
