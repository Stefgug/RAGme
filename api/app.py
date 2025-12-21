from fastapi import FastAPI
from langchain_community.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app = FastAPI()

# Initialize Ollama LLM (connects to your ollama container)
llm = Ollama(
    model="mistral:latest",
    base_url="http://localhost:11434"
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

# Initialize Qdrant retriever (connects to your qdrant container)
qdrant_client = QdrantClient("http://qdrant:6333")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="documents_collection",
    embedding=embeddings
)
retriever = vector_store.as_retriever()

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Example usage
@app.post("/ask")
def ask_question(question: str):
    result = qa_chain.invoke({"query": question})
    return {"answer": result["result"]}
