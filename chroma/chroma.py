import chromadb
from datetime import datetime
import uuid
import json
import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from chromadb.config import Settings
from common.models import Message
import os

print("Starting chroma client")

llm_model = "nomic-embed-text"

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

collection = chroma_client.get_or_create_collection(
    name="constitution",
    embedding_function=embedding
)



chat_collection = chroma_client.get_or_create_collection(
    name="chat_collection",
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



def initialize_collection(force_reload=False):
    """
    Initialize the collection with documents.
    Only loads if the collection is empty or if force_reload is True.
    
    Args:
        force_reload (bool): If True, will reload documents even if collection is not empty.
    """
    doc_count = collection.count()
    print(f"Collection contains {doc_count} documents before initialization")
    
    if doc_count > 0 and not force_reload:
        print(f"Collection already contains {doc_count} documents. Skipping initialization.")
        try:
            print("Sampling collection content...")
            sample = collection.get(limit=1)
            if sample and sample["documents"]:
                print(f"Sample document: {sample['documents'][0][:100]}...")
            else:
                print("No sample documents found despite positive count")
        except Exception as e:
            print(f"Error sampling collection: {str(e)}")
        return
    
    json_file = "constitution_docs.json"
    if not os.path.exists(json_file):
        print(f"Document file {json_file} not found. Skipping initialization.")
        return
    
    print(f"Opening {json_file}")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = data["documents"]
        metadatas = data["metadatas"]
        ids = data["ids"]
        
        print(f"Loaded {len(documents)} documents from JSON file")
        print(f"First document ID: {ids[0]}")
        print(f"First document content preview: {documents[0][:100]}...")
        
        if doc_count > 0 and force_reload:
            print("Clearing existing documents before reloading")
            try:
                collection.delete(where={})
                print("Existing documents cleared")
            except Exception as e:
                print(f"Error clearing documents: {str(e)}")
        
        add_documents_to_collection(documents, ids, metadatas)
        print(f"Added {len(documents)} documents to collection.")
        
        doc_count_after = collection.count()
        print(f"Collection contains {doc_count_after} documents after initialization")
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        import traceback
        traceback.print_exc()

def query_chromadb(query_text, n_results=5):
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
    chat_collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[msg_id]
    )

def load_chat_history(chat_id):
    results = chat_collection.get(
        where={"chat_id": chat_id}
    )

    metadatas = results.get("metadatas") or []
    documents = results.get("documents") or []

    if not metadatas or not documents:
        return []

    chat_items = [
        (meta, doc)
        for meta, doc in zip(metadatas, documents)
        if meta.get("type") != "session"
    ]

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
    chat_collection.add(
        documents=["Chat session created."],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )

def get_all_chat_sessions():
    """
    Retrieve all distinct chat session IDs stored in ChromaDB.
    """
    results = chat_collection.get(where={"type": "session"})

    chat_ids = []
    metadatas = results.get("metadatas") or []
    for meta in metadatas:
        chat_id = meta.get("chat_id")
        if chat_id:
            chat_ids.append(chat_id)

    return list(set(chat_ids))

initialize_collection()
