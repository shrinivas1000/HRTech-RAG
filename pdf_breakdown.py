from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "data/hr_test.pdf"

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)
    return docs

if __name__ == "__main__":
    docs = load_and_split_pdf("data/hr_test.pdf")
    for i, doc in enumerate(docs[:3]):
        print(f"Chunk {i+1}:\n{doc.page_content}\n---")
        


