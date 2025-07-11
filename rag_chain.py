from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pdf_breakdown import load_and_split_pdf
from dotenv import load_dotenv


def embed_and_store(pdf_path, persist_directory="vectorstore"):
    """Process a single PDF and add to existing vector database"""
    load_dotenv()

    docs = load_and_split_pdf(pdf_path)

  
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )
    
   
    vectordb.add_documents(docs)
    vectordb.persist()
    
    print(f"Added {len(docs)} chunks to ChromaDB in '{persist_directory}'.")
    return len(docs)


def create_new_vectorstore(pdf_path, persist_directory="vectorstore"):
    """Create a completely new vector database (overwrites existing)"""
    load_dotenv()

    docs = load_and_split_pdf(pdf_path)

    vectordb = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Created new ChromaDB with {len(docs)} chunks in '{persist_directory}'.")
    return len(docs)



