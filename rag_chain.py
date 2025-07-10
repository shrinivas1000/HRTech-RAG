from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pdf_breakdown import load_and_split_pdf
from dotenv import load_dotenv


def embed_and_store(pdf_path, persist_directory="vectorstore"):
    # Load environment variables (for OpenAI API key)
    load_dotenv()
    # Load and split the PDF into chunks
    docs = load_and_split_pdf(pdf_path)
    # Create embeddings for each chunk and store in ChromaDB
    vectordb = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Stored {len(docs)} chunks in ChromaDB in '{persist_directory}'.")

if __name__ == "__main__":
    embed_and_store("data/hr_test.pdf")
