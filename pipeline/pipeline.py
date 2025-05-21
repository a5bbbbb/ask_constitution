
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

