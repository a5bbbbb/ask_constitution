import chromadb
from datetime import datetime
import uuid
import json
import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from chromadb.config import Settings
from common.models import Message

print("Starting chroma client")

llm_model = "llama2"

chroma_client = chromadb.HttpClient(host="chroma", port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings


    def __call__(self, input):
        print(f"Embedding input: {input[:1]}... total: {len(input)}")
        if isinstance(input, str):
            input = [input]

        embeddings = self.langchain_embeddings.embed_documents(input)
        print("Embedding complete.")

        numpy_embeddings = [np.array(embedding) for embedding in embeddings]
        return numpy_embeddings



embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://host.docker.internal:11434"
    )
)

collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding
)

def add_documents_to_collection(documents, ids, metadatas, batch_size=5):
    print("Adding docs into the collections", flush=True)
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        print(f"Adding batch {i} to {i + len(batch_docs)}...", flush=True)

        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"Batch {i} to {i + len(batch_docs)} added", flush=True)



print("Opening json")
with open("constitution_docs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("got the data")
print(data["ids"])
documents = data["documents"][:1]
metadatas = data["metadatas"][:1]
ids = data["ids"][:1]

add_documents_to_collection(documents, ids, metadatas)
 
print("Added docs")


def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]



def store_chat_message(chat_id, role, content):
    msg_id = str(uuid.uuid4())
    metadata = {
        "chat_id": chat_id,
        "role": role,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "chat"
    }
    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[msg_id]
    )


def load_chat_history(chat_id):
    results = collection.get(
        where={"chat_id": chat_id}
    )

    metadatas = results.get("metadatas") or []
    documents = results.get("documents") or []

    if not metadatas or not documents:
        return []

    # Filter only items where type != "session" (i.e. actual chat messages)
    chat_items = [
        (meta, doc)
        for meta, doc in zip(metadatas, documents)
        if meta.get("type") != "session"
    ]

    # Sort by timestamp
    chat = sorted(
        chat_items,
        key=lambda x: x[0].get("timestamp") or datetime.min.isoformat()
    )

    return [Message(actor=("user" if meta["role"] == "user" else "ai"), payload=doc) for meta, doc in chat]



def store_chat_session(chat_id):
    """
    Store a metadata record that represents the chat session.
    """
    metadata = {
        "chat_id": chat_id,
        "type": "session",
        "timestamp": datetime.utcnow().isoformat()
    }
    collection.add(
        documents=["Chat session created."],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )



def get_all_chat_sessions():
    """
    Retrieve all distinct chat session IDs stored in ChromaDB.
    """
    results = collection.get(where={"type": "session"})

    chat_ids = []
    metadatas = results.get("metadatas") or []
    for meta in metadatas:
        chat_id = meta.get("chat_id")
        if chat_id:
            chat_ids.append(chat_id)

    return list(set(chat_ids))  # ensure uniqueness

