from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}
)
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("Connected to ChromaDB.")

try:
    db.delete_collection()
    print("Successfully cleared all data from ChromaDB!")
except Exception as e:
    print(f"Error clearing database: {e}")