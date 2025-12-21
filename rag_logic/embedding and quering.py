from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import requests

from tqdm import tqdm


documents = PyPDFLoader("data/sample_pdf.pdf").load()


# Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

embeddings_list = embeddings.embed_documents([doc.page_content for doc in documents])

# Store in Qdrant
qdrant_client = QdrantClient("localhost:6333")

try:
    collection = qdrant_client.get_collection("documents_collection")
    print(f"Collection already exists")
except:
    # Collection doesn't exist, create it
    qdrant_client.create_collection(
        collection_name="documents_collection",
        vectors_config=VectorParams(size=len(embeddings_list[0]), distance=Distance.COSINE)
    )
    print("Collection created successfully")

for idx, vector in tqdm(enumerate(embeddings_list), total=len(embeddings_list)):
    qdrant_client.upsert(
        collection_name="documents_collection",
        points=[
            {
                "id": idx,
                "vector": vector,
                "payload": {
                    "text": documents[idx].page_content
                }
            }
        ]
    )
print("Documents embedded and stored in Qdrant successfully")

# Example query to generate a response using the stored documents
text = qdrant_client.query_points(
    collection_name="documents_collection",
    query=embeddings.embed_query("What is this document about?"),
)


reponse = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model":"mistral:latest",
        "prompt": f"Question: What is the main topic of the document {text}?  \nAnswer:",
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
)

print("Response from the model:")
print(reponse.json()['response'])
