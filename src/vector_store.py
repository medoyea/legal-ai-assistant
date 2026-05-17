import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

VECTORSTORE_PATH = "vectorstore"


def create_vector_store(chunks: list[Document], embeddings) -> FAISS:
    
    print(f" Creating vector store from {len(chunks)} chunks...")
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vector_store.save_local(VECTORSTORE_PATH)
    
    print(f"Vector store created and saved to '{VECTORSTORE_PATH}/'")
    
    return vector_store


def load_vector_store(embeddings) -> FAISS | None:
    
    # Check if a saved vector store exists
# Check if the actual FAISS index files exist inside the folder
    index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(index_file):
        print("⚠️  No saved vector store found. Please upload documents first.")
        return None
    
    print("Loading vector store")
    
    vector_store = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print("Vector store loaded successfully!")
    
    return vector_store


def add_to_vector_store(existing_store: FAISS, new_chunks: list[Document], embeddings) -> FAISS:
    print(f"Adding {len(new_chunks)} new chunks to existing vector store...")
    
    # Create a new temporary store for the new chunks
    new_store = FAISS.from_documents(new_chunks, embeddings)
    
    # Merge the new store into the existing one
    existing_store.merge_from(new_store)
    
    # Save the updated store to disk
    existing_store.save_local(VECTORSTORE_PATH)
    
    print("Vector store updated and saved!")
    
    return existing_store


def search_vector_store(vector_store: FAISS, query: str, top_k: int = 4) -> list[Document]:

    relevant_chunks = vector_store.similarity_search(
        query=query,
        k=top_k
    )
    
    print(f"Found {len(relevant_chunks)} relevant chunks for the query")
    
    return relevant_chunks


def delete_vector_store():
    import shutil
    
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        print("Vector store deleted.")
    else:
        print("No vector store found to delete.")
