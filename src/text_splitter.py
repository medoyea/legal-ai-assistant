import os
import sys

# LangChain's text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain's Document class
from langchain.schema import Document


def split_pages_into_chunks(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators define where to split
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    raw_documents = []
    
    for page in pages:
        doc = Document(
            page_content=page["text"],
            metadata={
                "source": page["source"],
                "page_number": page["page_number"]
            }
        )
        raw_documents.append(doc)

    chunks = splitter.split_documents(raw_documents)
    
    print(f"Split {len(raw_documents)} pages into {len(chunks)} chunks")
    print(f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    
    return chunks


def preview_chunks(chunks: list[Document], num_to_show: int = 3):
    print(f"\n--- Preview of first {num_to_show} chunks ---")
    
    for i, chunk in enumerate(chunks[:num_to_show]):
        print(f"\nChunk #{i+1}")
        print(f"Source: {chunk.metadata.get('source', 'unknown')} | Page: {chunk.metadata.get('page_number', '?')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Text preview: {chunk.page_content[:200]}...")
        print("-" * 50)
