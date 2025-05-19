import chromadb
import json
import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from chromadb.config import Settings

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


# documents = [
#     "Mars, often called the 'Red Planet', has captured the imagination of scientists and space enthusiasts alike.",
#     "The Hubble Space Telescope has provided us with breathtaking images of distant galaxies and nebulae.",
#     "The concept of a black hole, where gravity is so strong that nothing can escape it, was first theorized by Albert Einstein's theory of general relativity.",
#     "The Renaissance was a pivotal period in history that saw a flourishing of art, science, and culture in Europe.",
#     "The Industrial Revolution marked a significant shift in human society, leading to urbanization and technological advancements.",
#     "The ancient city of Rome was once the center of a powerful empire that spanned across three continents.",
#     "Dolphins are known for their high intelligence and social behavior, often displaying playful interactions with humans.",
#     "The chameleon is a remarkable creature that can change its skin color to blend into its surroundings or communicate with other chameleons.",
#     "The migration of monarch butterflies spans thousands of miles and involves multiple generations to complete.",
#     "Christopher Nolan's 'Inception' is a mind-bending movie that explores the boundaries of reality and dreams.",
#     "The 'Lord of the Rings' trilogy, directed by Peter Jackson, brought J.R.R. Tolkien's epic fantasy world to life on the big screen.",
#     "Pixar's 'Toy Story' was the first feature-length film entirely animated using computer-generated imagery (CGI).",
#     "Superman, known for his incredible strength and ability to fly, is one of the most iconic superheroes in comic book history.",
#     "Black Widow, portrayed by Scarlett Johansson, is a skilled spy and assassin in the Marvel Cinematic Universe.",
#     "The character of Iron Man, played by Robert Downey Jr., kickstarted the immensely successful Marvel movie franchise in 2008."
# ]
# metadatas = [{'source': "Space"}, {'source': "Space"}, {'source': "Space"}, {'source': "History"}, {'source': "History"}, {'source': "History"}, {'source': "Animals"}, {'source': "Animals"}, {'source': "Animals"}, {'source': "Movies"}, {'source': "Movies"}, {'source': "Movies"}, {'source': "Superheroes"}, {'source': "Superheroes"}, {'source': "Superheroes"}]
# ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]



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







