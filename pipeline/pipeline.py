
"""
Pipeline module for the RAG system.
"""
import os
from ollama_client.llm import query_ollama
from chroma.chroma import query_constitution_documents 
import requests
import json




def fetch_docs():
    try:
        response = requests.get("http://host.docker.internal:6660/embeddings")
        response.raise_for_status()
        
        api_data = response.json()
        retrieved_ids = api_data.get("ids", [])
        
        if not retrieved_ids:
            print("No document IDs retrieved from API")
            return []
        
        print(f"Retrieved IDs from API: {retrieved_ids}")
        
        json_file = "constitution_docs.json"
        if not os.path.exists(json_file):
            print(f"Document file {json_file} not found. Skipping initialization.")
            return []
        
        print(f"Opening {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = data["documents"]
        metadatas = data["metadatas"]
        ids = data["ids"]
        
        filtered_docs = []
        for i, doc_id in enumerate(ids):
            if doc_id in retrieved_ids:
                doc_info = {
                    "id": doc_id,
                    "document": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else None
                }
                filtered_docs.append(doc_info)
        
        print(f"Found {len(filtered_docs)} matching documents out of {len(retrieved_ids)} requested IDs")
        return filtered_docs
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from API: {e}")
        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        import traceback
        traceback.print_exc()
        return []


# def fetch_docs():
#     #["1","2"]
#     response = requests.get("http://host.docker.internal:6660/embeddings")
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
#
#     except Exception as e:
#         print(f"Error loading documents: {e}")
#         import traceback
#         traceback.print_exc()
#






def rag_pipeline(query_text, chat_history=None):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB, chat history, and Ollama.
    
    Args:
        query_text (str): The input query.
        chat_history (list, optional): List of previous Message objects from the conversation.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    try:
        print("Query: " + query_text)
        
        # retrieved_docs, metadata = query_chromadb(query_text)
        # constitution_context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        

        retrieved_docs = query_constitution_documents(query_text, n_results=5)
        constitution_context = " ".join(retrieved_docs) if retrieved_docs else "No relevant documents found."


        print("Constitution Context: " + constitution_context[:200] + "..." if len(constitution_context) > 200 else constitution_context)
        
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            recent_messages = chat_history[-10:] if len(chat_history) > 10 else chat_history
            
            chat_context = "\nPrevious conversation:\n"
            for msg in recent_messages:
                prefix = "Human: " if msg.actor == "user" else "AI: "
                chat_context += f"{prefix}{msg.payload}\n"
                
            print("Chat Context: " + chat_context[:200] + "..." if len(chat_context) > 200 else chat_context)
        
        augmented_prompt = f"""
Constitution Context: {constitution_context}
{chat_context}

Current Question: {query_text}

Answer the current question based primarily on the Constitution Context above.
Maintain a consistent and helpful tone, taking into account the previous conversation if relevant.
If the Constitution Context doesn't provide enough information to answer fully, 
make that clear but still give the best possible answer based on available information.
"""
        
        print("######## Augmented Prompt ########")
        print(augmented_prompt)
        
        response = query_ollama(augmented_prompt)
        return response
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return f"I'm sorry, there was an error processing your query: {str(e)}"

