import requests
from datetime import datetime
import uuid
import json
import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from common.models import Message
import os

print("Starting blockchain embedding client")

llm_model = "nomic-embed-text"

class BlockchainEmbeddingClient:
    """
    Client for storing and retrieving embeddings from blockchain via API.
    Stores documents locally and only embeddings on blockchain for retrieval.
    """
    def __init__(self, api_base_url="http://host.docker.internal:6660", embedding_model=None):
        self.api_base_url = api_base_url
        self.embeddings_endpoint = f"{api_base_url}/embeddings"
        
        self.documents_store = {}
        self.metadata_store = {}
        
        if embedding_model is None:
            self.embedding_model = OllamaEmbeddings(
                model=llm_model,
                base_url="http://host.docker.internal:11434"
            )
        else:
            self.embedding_model = embedding_model
    
    def compute_embeddings(self, texts):
        """
        Compute embeddings using local Ollama model.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.embed_documents(texts)
        print("Embedding computation complete.")
        return embeddings
    
    def store_embeddings(self, embeddings, ids, documents=None, metadatas=None):
        """
        Store embeddings in blockchain via API and documents locally.
        
        Args:
            embeddings: List of embedding vectors
            ids: List of document IDs (must be integers for blockchain)
            documents: List of document texts (stored locally)
            metadatas: List of metadata dicts (stored locally)
        """
        try:
            if isinstance(embeddings[0], np.ndarray):
                embeddings = [emb.tolist() for emb in embeddings]
            
            int_ids = [int(id_val) for id_val in ids]
            
            if documents:
                for i, doc_id in enumerate(int_ids):
                    self.documents_store[doc_id] = documents[i]
            
            if metadatas:
                for i, doc_id in enumerate(int_ids):
                    self.metadata_store[doc_id] = metadatas[i]
            
            payload = {
                "embeddings": embeddings,
                "ids": int_ids
            }
            
            response = requests.post(self.embeddings_endpoint, json=payload)
            response.raise_for_status()
            
            print(f"Successfully stored {len(embeddings)} embeddings in blockchain")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error storing embeddings in blockchain: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
        except ValueError as e:
            print(f"Error converting IDs to integers: {e}")
            raise
    
    def query_similar_documents(self, query_text, n_results=5):
        """
        Query blockchain for similar documents using text input.
        
        Args:
            query_text: Text to find similar documents for
            n_results: Number of results to return (not used in current API)
        
        Returns:
            list: List of document dictionaries with id, document, and metadata
        """
        try:
            # Compute embedding for query text
            query_embedding = self.compute_embeddings([query_text])[0]
            
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Query blockchain API
            payload = {
                "embeddings": [query_embedding]
            }
            
            response = requests.get(self.embeddings_endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            retrieved_ids = result.get("ids", [])
            
            print(f"Blockchain returned {len(retrieved_ids)} similar document IDs: {retrieved_ids}")
            
            # Get documents from local storage
            documents = []
            for doc_id in retrieved_ids:
                doc_info = {
                    "id": doc_id,
                    "document": self.documents_store.get(doc_id, f"Document {doc_id} not found locally"),
                    "metadata": self.metadata_store.get(doc_id, {})
                }
                documents.append(doc_info)
            
            return documents
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying blockchain: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return []
        except Exception as e:
            print(f"Error in query_similar_documents: {e}")
            return []
    
    def get_documents_by_ids(self, doc_ids):
        """
        Get documents by their IDs from local storage.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            list: List of document dictionaries
        """
        documents = []
        for doc_id in doc_ids:
            int_id = int(doc_id) if not isinstance(doc_id, int) else doc_id
            doc_info = {
                "id": int_id,
                "document": self.documents_store.get(int_id, f"Document {int_id} not found"),
                "metadata": self.metadata_store.get(int_id, {})
            }
            documents.append(doc_info)
        
        return documents

# Initialize the blockchain client
blockchain_client = BlockchainEmbeddingClient()

def add_documents_to_blockchain(documents, ids, metadatas, batch_size=5):
    """
    Add documents to blockchain in batches.
    """
    print("Adding docs to blockchain", flush=True)
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
        
        print(f"Processing batch {i} to {i + len(batch_docs)}...", flush=True)
        
        # Compute embeddings for this batch
        embeddings = blockchain_client.compute_embeddings(batch_docs)
        
        # Store in blockchain (embeddings) and locally (documents + metadata)
        blockchain_client.store_embeddings(
            embeddings=embeddings,
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadatas
        )
        
        print(f"Batch {i} to {i + len(batch_docs)} added to blockchain", flush=True)

def initialize_blockchain_collection(force_reload=False):
    """
    Initialize the blockchain collection with documents from JSON file.
    """
    try:
        # Check if we already have documents loaded locally
        existing_count = len(blockchain_client.documents_store)
        
        print(f"Local storage contains {existing_count} documents before initialization")
        
        if existing_count > 0 and not force_reload:
            print(f"Documents already loaded locally. Skipping initialization.")
            return
        
        json_file = "constitution_docs.json"
        if not os.path.exists(json_file):
            print(f"Document file {json_file} not found. Skipping initialization.")
            return
        
        print(f"Opening {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = data["documents"]
        metadatas = data["metadatas"]
        ids = data["ids"]
        
        print(f"Loaded {len(documents)} documents from JSON file")
        print(f"First document ID: {ids[0]}")
        print(f"First document content preview: {documents[0][:100]}...")
        
        if force_reload:
            print("Force reload: clearing local storage")
            blockchain_client.documents_store.clear()
            blockchain_client.metadata_store.clear()
        
        add_documents_to_blockchain(documents, ids, metadatas)
        print(f"Added {len(documents)} documents to blockchain and local storage.")
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        import traceback
        traceback.print_exc()

def query_constitution_documents(query_text, n_results=5):
    """
    Main method to query constitution documents using the blockchain.
    This is the method your pipeline should call.
    
    Args:
        query_text (str): The input query/prompt.
        n_results (int): Number of results (currently not used by blockchain API).
    
    Returns:
        list: List of relevant documents with id, document text, and metadata.
    """
    print(f"Querying constitution documents for: '{query_text[:50]}...'")
    
    documents = blockchain_client.query_similar_documents(query_text, n_results)
    
    print(f"Found {len(documents)} relevant documents")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. ID: {doc['id']} - {doc['document'][:80]}...")
    
    return documents

def get_documents_by_ids(doc_ids):
    """
    Helper method to get specific documents by their IDs.
    
    Args:
        doc_ids: List of document IDs
        
    Returns:
        list: List of document dictionaries
    """
    return blockchain_client.get_documents_by_ids(doc_ids)

# Chat functionality (simplified - storing locally since blockchain is for constitution docs)
chat_storage = {}  # chat_id -> list of messages

def store_chat_message(chat_id, role, content):
    """
    Store chat message locally (since blockchain is for constitution documents).
    """
    if chat_id not in chat_storage:
        chat_storage[chat_id] = []
    
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    chat_storage[chat_id].append(message)
    print(f"Stored {role} message in chat {chat_id}")

def load_chat_history(chat_id):
    """
    Load chat history from local storage.
    """
    messages = chat_storage.get(chat_id, [])
    
    return [Message(
        actor=("user" if msg["role"] == "user" else "ai"),
        payload=msg["content"]
    ) for msg in sorted(messages, key=lambda x: x["timestamp"])]

def store_chat_session(chat_id):
    """
    Initialize a new chat session.
    """
    if chat_id not in chat_storage:
        chat_storage[chat_id] = []
    print(f"Initialized chat session: {chat_id}")

def get_all_chat_sessions():
    """
    Get all chat session IDs.
    """
    return list(chat_storage.keys())

print("Initializing constitution documents in blockchain...")
initialize_blockchain_collection()
print("Blockchain client ready!")





# import chromadb
# from datetime import datetime
# import uuid
# import json
# import numpy as np
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from chromadb.config import Settings
# from common.models import Message
# import os
#
# print("Starting chroma client")
#
# llm_model = "nomic-embed-text"
#
# chroma_client = chromadb.HttpClient(host="chroma", port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
#
# class ChromaDBEmbeddingFunction:
#     """
#     Custom embedding function for ChromaDB using embeddings from Ollama.
#     """
#     def __init__(self, langchain_embeddings):
#         self.langchain_embeddings = langchain_embeddings
#
#     def __call__(self, input):
#         print(f"Embedding input: {input[:1]}... total: {len(input)}")
#         if isinstance(input, str):
#             input = [input]
#
#         embeddings = self.langchain_embeddings.embed_documents(input)
#         print("Embedding complete.")
#
#         numpy_embeddings = [np.array(embedding) for embedding in embeddings]
#         return numpy_embeddings
#
# embedding = ChromaDBEmbeddingFunction(
#     OllamaEmbeddings(
#         model=llm_model,
#         base_url="http://host.docker.internal:11434"
#     )
# )
#
# collection = chroma_client.get_or_create_collection(
#     name="constitution",
#     embedding_function=embedding
# )
#
#
#
# chat_collection = chroma_client.get_or_create_collection(
#     name="chat_collection",
#     embedding_function=embedding
# )
#
#
# def add_documents_to_collection(documents, ids, metadatas, batch_size=5):
#     print("Adding docs into the collections", flush=True)
#     for i in range(0, len(documents), batch_size):
#         batch_docs = documents[i:i + batch_size]
#         batch_ids = ids[i:i + batch_size]
#         batch_metadatas = metadatas[i:i + batch_size]
#         print(f"Adding batch {i} to {i + len(batch_docs)}...", flush=True)
#
#         collection.add(
#             documents=batch_docs,
#             metadatas=batch_metadatas,
#             ids=batch_ids
#         )
#         print(f"Batch {i} to {i + len(batch_docs)} added", flush=True)
#
#
#
# def initialize_collection(force_reload=False):
#     """
#     Initialize the collection with documents.
#     Only loads if the collection is empty or if force_reload is True.
#     
#     Args:
#         force_reload (bool): If True, will reload documents even if collection is not empty.
#     """
#     doc_count = collection.count()
#     print(f"Collection contains {doc_count} documents before initialization")
#     
#     if doc_count > 0 and not force_reload:
#         print(f"Collection already contains {doc_count} documents. Skipping initialization.")
#         try:
#             print("Sampling collection content...")
#             sample = collection.get(limit=1)
#             if sample and sample["documents"]:
#                 print(f"Sample document: {sample['documents'][0][:100]}...")
#             else:
#                 print("No sample documents found despite positive count")
#         except Exception as e:
#             print(f"Error sampling collection: {str(e)}")
#         return
#     
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
#         print(f"Loaded {len(documents)} documents from JSON file")
#         print(f"First document ID: {ids[0]}")
#         print(f"First document content preview: {documents[0][:100]}...")
#         
#         if doc_count > 0 and force_reload:
#             print("Clearing existing documents before reloading")
#             try:
#                 collection.delete(where={})
#                 print("Existing documents cleared")
#             except Exception as e:
#                 print(f"Error clearing documents: {str(e)}")
#         
#         add_documents_to_collection(documents, ids, metadatas)
#         print(f"Added {len(documents)} documents to collection.")
#         
#         doc_count_after = collection.count()
#         print(f"Collection contains {doc_count_after} documents after initialization")
#         
#     except Exception as e:
#         print(f"Error loading documents: {e}")
#         import traceback
#         traceback.print_exc()
#
# def query_chromadb(query_text, n_results=5):
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
#     results = collection.query(
#         query_texts=[query_text],
#         n_results=n_results
#     )
#     return results["documents"], results["metadatas"]
#
#
#
#
#
# def store_chat_message(chat_id, role, content):
#     msg_id = str(uuid.uuid4())
#     metadata = {
#         "chat_id": chat_id,
#         "role": role,
#         "timestamp": datetime.utcnow().isoformat(),
#         "type": "chat"
#     }
#     chat_collection.add(
#         documents=[content],
#         metadatas=[metadata],
#         ids=[msg_id]
#     )
#
# def load_chat_history(chat_id):
#     results = chat_collection.get(
#         where={"chat_id": chat_id}
#     )
#
#     metadatas = results.get("metadatas") or []
#     documents = results.get("documents") or []
#
#     if not metadatas or not documents:
#         return []
#
#     chat_items = [
#         (meta, doc)
#         for meta, doc in zip(metadatas, documents)
#         if meta.get("type") != "session"
#     ]
#
#     chat = sorted(
#         chat_items,
#         key=lambda x: x[0].get("timestamp") or datetime.min.isoformat()
#     )
#
#     return [Message(actor=("user" if meta["role"] == "user" else "ai"), payload=doc) for meta, doc in chat]
#
# def store_chat_session(chat_id):
#     """
#     Store a metadata record that represents the chat session.
#     """
#     metadata = {
#         "chat_id": chat_id,
#         "type": "session",
#         "timestamp": datetime.utcnow().isoformat()
#     }
#     chat_collection.add(
#         documents=["Chat session created."],
#         metadatas=[metadata],
#         ids=[str(uuid.uuid4())]
#     )
#
# def get_all_chat_sessions():
#     """
#     Retrieve all distinct chat session IDs stored in ChromaDB.
#     """
#     results = chat_collection.get(where={"type": "session"})
#
#     chat_ids = []
#     metadatas = results.get("metadatas") or []
#     for meta in metadatas:
#         chat_id = meta.get("chat_id")
#         if chat_id:
#             chat_ids.append(chat_id)
#
#     return list(set(chat_ids))
#
# initialize_collection()
