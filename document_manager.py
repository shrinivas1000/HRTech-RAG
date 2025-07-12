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
import logging

class DocumentManager:
    def __init__(self, vectorstore_path: str = "vectorstore"):
        self.vectorstore_path = vectorstore_path
        self.uploaded_docs: List[Dict] = []
        
    async def process_uploaded_file(self, file: UploadFile) -> Dict:
        """Process uploaded PDF and add to vector database"""
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            # Get file size before processing (while file still exists)
            file_size = os.path.getsize(temp_file_path)
        
        try:
            chunk_count = self._add_to_vectorstore(temp_file_path, file.filename)
            
            # Safer metadata creation with explicit type conversion
            doc_info = {
                "filename": str(file.filename),
                "upload_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "chunk_count": int(chunk_count),
                "file_size": int(file_size)  # Use pre-calculated size
            }
            
            # Add to tracking with error handling
            try:
                self.uploaded_docs.append(doc_info)
            except Exception as e:
                logging.warning(f"Failed to track document: {e}")
                # Continue anyway - processing succeeded
            
            return doc_info
            
        except Exception as e:
            logging.error(f"Document processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        finally:
            # Clean up temp file with error handling
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file: {e}")
    
    def _add_to_vectorstore(self, file_path: str, filename: str) -> int:
        """Add document to existing vector database"""
        
        try:
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
                        "upload_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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

            
            return len(documents)
            
        except Exception as e:
            logging.error(f"Vector store operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")
    
    def get_document_stats(self) -> Dict:
        """Get statistics about uploaded documents"""
        try:
            total_docs = len(self.uploaded_docs)
            total_chunks = sum(doc.get("chunk_count", 0) for doc in self.uploaded_docs)
            total_size = sum(doc.get("file_size", 0) for doc in self.uploaded_docs)
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_size_bytes": total_size,
                "documents": self.uploaded_docs
            }
        except Exception as e:
            logging.error(f"Failed to get document stats: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_size_bytes": 0,
                "documents": []
            }
