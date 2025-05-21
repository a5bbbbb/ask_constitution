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

# def initialize_collection(force_reload=False):
#     """
#     Initialize the collection with documents.
#     Only loads if the collection is empty or if force_reload is True.
#     
#     Args:
#         force_reload (bool): If True, will reload documents even if collection is not empty.
#     """
#     # Check if documents already exist in the collection
#     doc_count = collection.count()
#     
#     # Skip loading if documents already exist and force_reload is False
#     if doc_count > 0 and not force_reload:
#         print(f"Collection already contains {doc_count} documents. Skipping initialization.")
#         return
#     
#     # Load document file
#     json_file = "constitution_docs.json"
#     if not os.path.exists(json_file):
#         print(f"Document file {json_file} not found. Skipping initialization.")
#         return
#     
#     print(f"Opening {json_file}")
#     try:
#         with open(json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         
#         documents = data["documents"]
#         metadatas = data["metadatas"]
#         ids = data["ids"]
#         
#         add_documents_to_collection(documents, ids, metadatas)
#         print(f"Added {len(documents)} documents to collection.")
#     except Exception as e:
#         print(f"Error loading documents: {e}")



def initialize_collection(force_reload=False):
    """
    Initialize the collection with documents.
    Only loads if the collection is empty or if force_reload is True.
    
    Args:
        force_reload (bool): If True, will reload documents even if collection is not empty.
    """
    # Check if documents already exist in the collection
    doc_count = collection.count()
    print(f"Collection contains {doc_count} documents before initialization")
    
    # Skip loading if documents already exist and force_reload is False
    if doc_count > 0 and not force_reload:
        print(f"Collection already contains {doc_count} documents. Skipping initialization.")
        # Debug: Check what's in the collection
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
    
    # Load document file
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
        
        # Clear existing documents if needed
        if doc_count > 0 and force_reload:
            print("Clearing existing documents before reloading")
            try:
                collection.delete(where={})
                print("Existing documents cleared")
            except Exception as e:
                print(f"Error clearing documents: {str(e)}")
        
        add_documents_to_collection(documents, ids, metadatas)
        print(f"Added {len(documents)} documents to collection.")
        
        # Verify the documents were added
        doc_count_after = collection.count()
        print(f"Collection contains {doc_count_after} documents after initialization")
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        import traceback
        traceback.print_exc()

#Query functions (these don't load documents, only query the DB)
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



# def query_chromadb(query_text, n_results=3):  # Increase default n_results
#     """
#     Query the ChromaDB collection for relevant documents.
#     
#     Args:
#         query_text (str): The input query.
#         n_results (int): The number of top results to return.
#     
#     Returns:
#         list of dict: The top matching documents and their metadata.
#     """
#     print(f"ChromaDB Query: '{query_text}'")
#     
#     # First check if we have a direct article reference in the query
#     import re
#     article_match = re.search(r'article\s*(\d+)', query_text.lower())
#     
#     if article_match:
#         article_num = article_match.group(1)
#         print(f"Detected reference to Article {article_num}, trying direct retrieval")
#         
#         # Try direct retrieval by ID first
#         try:
#             article_result = collection.get(ids=[article_num])
#             if article_result and article_result["documents"] and article_result["documents"][0]:
#                 print(f"Successfully retrieved Article {article_num} by ID")
#                 return [article_result["documents"]], [article_result["metadatas"]]
#         except Exception as e:
#             print(f"Error retrieving Article {article_num} by ID: {str(e)}")
#     
#     # If direct retrieval failed or no article number was specified, fall back to semantic search
#     print(f"Performing semantic search for: '{query_text}'")
#     
#     # Get document count
#     doc_count = collection.count()
#     print(f"Collection has {doc_count} documents")
#     
#     if doc_count == 0:
#         print("WARNING: ChromaDB collection is empty!")
#         # Try to initialize the collection
#         try:
#             from chroma.chroma import initialize_collection
#             print("Attempting to initialize collection...")
#             initialize_collection(force_reload=True)
#             doc_count = collection.count()
#             print(f"After initialization, collection has {doc_count} documents")
#         except Exception as e:
#             print(f"Failed to initialize collection: {str(e)}")
#     
#     # Perform the query
#     try:
#         results = collection.query(
#             query_texts=[query_text],
#             n_results=n_results
#         )
#         
#         if not results or not results["documents"] or not results["documents"][0]:
#             print("No results returned from ChromaDB query")
#             return [["No relevant documents found."]], [{}]
#         
#         print(f"ChromaDB returned {len(results['documents'][0])} documents")
#         
#         # Debug: check if the results actually contain meaningful content
#         for i, doc in enumerate(results["documents"][0][:2]):
#             print(f"Result {i+1} preview: {doc[:100]}...")
#         
#         # Check if results contain just the query text (which indicates a problem)
#         if len(results["documents"][0]) == 1 and results["documents"][0][0] == query_text:
#             print("WARNING: ChromaDB returned only the query text, not actual documents!")
#             
#             # Fallback: try to retrieve any document
#             print("Attempting to retrieve a sample document...")
#             try:
#                 sample = collection.get(limit=5)
#                 if sample and sample["documents"]:
#                     print(f"Retrieved {len(sample['documents'])} sample documents")
#                     return [sample["documents"]], [sample["metadatas"]]
#                 else:
#                     print("No documents found in the collection")
#             except Exception as e:
#                 print(f"Error retrieving sample: {str(e)}")
#                 
#             # If all fails, return a message
#             return [["Unable to retrieve relevant information from the database."]], [{}]
#             
#         return results["documents"], results["metadatas"]
#         
#     except Exception as e:
#         print(f"Error during ChromaDB query: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return [["Error querying the database."]], [{}]





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

    return list(set(chat_ids))  # ensure uniqueness

# Uncomment this line if you want this module to load documents when imported
initialize_collection()
