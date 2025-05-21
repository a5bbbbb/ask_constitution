
"""
Pipeline module for the RAG system.
"""
from ollama_client.llm import query_ollama
from chroma.chroma import query_chromadb






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
        
        retrieved_docs, metadata = query_chromadb(query_text)
        constitution_context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        
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

# def rag_pipeline(query_text, chat_context):
#     """
#     Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
#     
#     Args:
#         query_text (str): The input query.
#     
#     Returns:
#         str: The generated response from Ollama augmented with retrieved context.
#     """
#     try:
#         print("Query: "+query_text)
#         retrieved_docs, metadata = query_chromadb(query_text)
#         
#         if not retrieved_docs or not retrieved_docs[0]:
#             print("No documents retrieved from ChromaDB")
#             context = "No relevant documents found in the Constitution of Kazakhstan."
#         else:
#             if len(retrieved_docs[0]) == 1 and retrieved_docs[0][0] == query_text:
#                 print("WARNING: ChromaDB returned the query itself, not actual documents!")
#                 context = "Unable to retrieve specific information from the Constitution of Kazakhstan."
#             else:
#                 context = " ".join(retrieved_docs[0]).join(" ").join(chat_context)
#                 print(f"Successfully retrieved {len(retrieved_docs[0])} relevant documents")
#         
#         print("Context: "+ context[:200] + "..." if len(context) > 200 else context)
#         
#         augmented_prompt = (
#             f"Context: {context}\n\n"
#             f"Question: {query_text}\n\n"
#             "Answer the question based only on the information in the context above. "
#             "If the context doesn't provide enough information to answer the question, "
#             "say that you don't have sufficient information from the Constitution of Kazakhstan to answer."
#         )
#         
#         print("######## Augmented Prompt ########")
#         print(augmented_prompt)
#         
#         response = query_ollama(augmented_prompt)
#         return response
#     except Exception as e:
#         print(f"Error in RAG pipeline: {e}")
#         import traceback
#         traceback.print_exc()
#         return f"I'm sorry, there was an error processing your query: {str(e)}"
#
#
#
