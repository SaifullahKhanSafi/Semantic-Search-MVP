from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# Configuration
DB_DIR = "./chroma_db"

def run_search():
    print("Loading embedding model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    print(f"Connecting to vector database at '{DB_DIR}'...")
    try:
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        print("Connected successfully!\n")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

   
    print("Loading Cross-Encoder Re-ranker...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    print("==================================================")
    print("Semantic Search is active. Type 'exit' to stop.")
    print("==================================================")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() in ['exit', 'quit']:
            print("Exiting search...")
            break

        if not query.strip():
            continue

        print("Searching...")

        # Step 1: Broad vector search
        initial_results = db.similarity_search(query, k=10)

        if not initial_results:
            print("No relevant results found.")
            continue

        # Step 2: Re-rank with cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_results]
        scores = cross_encoder.predict(pairs)
        sorted_results = sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)
        top_results = sorted_results[:3]

        for i, (score, result) in enumerate(top_results):
            print(f"\n--- Top Match {i+1} (score: {score:.4f}) ---")
            print(f"Source: {result.metadata.get('source', 'Unknown')}")
            print(result.page_content)

if __name__ == "__main__":
    run_search()