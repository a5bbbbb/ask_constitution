
"""
Pipeline module for the RAG system.
"""
from ollama_client.llm import query_ollama
from chroma.chroma import query_chromadb

# def rag_pipeline(query_text):
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
#         # print("Docs: "+retrieved_docs[:1])
#         context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
#         print("Contect: "+ context)
#
#         # Step 2: Send the query along with the context to Ollama
#         augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
#         print("######## Augmented Prompt ########")
#         print(augmented_prompt)
#
#         response = query_ollama(augmented_prompt)
#         return response
#     except Exception as e:
#         print(f"Error in RAG pipeline: {e}")
#         return f"I'm sorry, there was an error processing your query: {str(e)}"



def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    try:
        print("Query: "+query_text)
        retrieved_docs, metadata = query_chromadb(query_text)
        
        # Validate the retrieved docs
        if not retrieved_docs or not retrieved_docs[0]:
            print("No documents retrieved from ChromaDB")
            context = "No relevant documents found in the Constitution of Kazakhstan."
        else:
            # Check if we just got the query back
            if len(retrieved_docs[0]) == 1 and retrieved_docs[0][0] == query_text:
                print("WARNING: ChromaDB returned the query itself, not actual documents!")
                context = "Unable to retrieve specific information from the Constitution of Kazakhstan."
            else:
                # Join the retrieved documents
                context = " ".join(retrieved_docs[0])
                print(f"Successfully retrieved {len(retrieved_docs[0])} relevant documents")
        
        print("Context: "+ context[:200] + "..." if len(context) > 200 else context)
        
        # Send the query along with the context to Ollama
        augmented_prompt = (
            f"Context: {context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer the question based only on the information in the context above. "
            "If the context doesn't provide enough information to answer the question, "
            "say that you don't have sufficient information from the Constitution of Kazakhstan to answer."
        )
        
        print("######## Augmented Prompt ########")
        print(augmented_prompt)
        
        response = query_ollama(augmented_prompt)
        return response
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return f"I'm sorry, there was an error processing your query: {str(e)}"



