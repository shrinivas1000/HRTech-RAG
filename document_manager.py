from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import tempfile
import os
import datetime
from pathlib import Path
from typing import List, Dict

class DocumentManager:
    def __init__(self, vectorstore_path: str = "vectorstore"):
        self.vectorstore_path = vectorstore_path
        self.uploaded_docs: List[Dict] = []
        
    async def process_uploaded_file(self, file: UploadFile) -> Dict:  # Added async
        """Process uploaded PDF and add to vector database"""
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()  # Fixed file reading
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            chunk_count = self._add_to_vectorstore(temp_file_path, file.filename)
            
            doc_info = {
                "filename": file.filename,
                "upload_date": datetime.datetime.now().isoformat(),
                "chunk_count": chunk_count,
                "file_size": os.path.getsize(temp_file_path)
            }
            self.uploaded_docs.append(doc_info)
            
            return doc_info
            
        finally:
            os.unlink(temp_file_path)
    
    def _add_to_vectorstore(self, file_path: str, filename: str) -> int:
        """Add document to existing vector database"""
        
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        documents = []
        for page in pages:
            chunks = text_splitter.split_text(page.page_content)
            for chunk in chunks:
                chunk_metadata = {
                    "page": page.metadata.get("page", 0),
                    "source": filename,
                    "upload_date": datetime.datetime.now().isoformat()
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        vectordb = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=OpenAIEmbeddings()
        )
        
        vectordb.add_documents(documents)
        vectordb.persist()
        
        return len(documents)
    
    def get_document_stats(self) -> Dict:
        """Get statistics about uploaded documents"""
        total_docs = len(self.uploaded_docs)
        total_chunks = sum(doc.get("chunk_count", 0) for doc in self.uploaded_docs)
        total_size = sum(doc.get("file_size", 0) for doc in self.uploaded_docs)
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "documents": self.uploaded_docs
        }
