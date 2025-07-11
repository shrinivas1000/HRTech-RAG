from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path):
    """Load and split PDF into chunks for vector storage"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    docs = splitter.split_documents(pages)
    return docs

def get_pdf_info(pdf_path):
    """Get basic information about a PDF file"""
    docs = load_and_split_pdf(pdf_path)
    
    return {
        "total_chunks": len(docs),
        "total_pages": len(set(doc.metadata.get("page", 0) for doc in docs)),
        "sample_content": docs[0].page_content[:200] + "..." if docs else "No content"
    }


