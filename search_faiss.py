import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("property_index.faiss")

# Load property data
with open("filtered_properties.json", "r", encoding="utf-8") as f:
    properties = json.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# User query
query = input("\nEnter property search query: ")

# Convert query to vector
query_vector = model.encode([query]).astype(np.float32)

# Search FAISS for top 5 matches
D, I = index.search(query_vector, k=5)

# Fetch matching properties
matching_properties = [properties[i] for i in I[0] if i < len(properties)]

# Remove properties with "N/A" agent info
for prop in matching_properties:
    if prop["agent_name"] == "N/A":
        del prop["agent_name"]
        del prop["agent_contact"]
        del prop["agent_email"]

# Display results
if matching_properties:
    print("\n🔹 **Top Matching Properties:**\n")
    for prop in matching_properties:
        print(json.dumps(prop, indent=4))
else:
    print("\n❌ No matching properties found. Try a different search query.")
