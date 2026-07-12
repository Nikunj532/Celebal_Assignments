import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import glob as glob_module
import pypdf
import docx

def load_all_files(folder_path):
    docs = []

    # .txt files
    for filepath in glob_module.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({"content": content, "source": filepath})

    # .pdf files
    for filepath in glob_module.glob(os.path.join(folder_path, "*.pdf")):
        reader = pypdf.PdfReader(filepath)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
        docs.append({"content": content, "source": filepath})

    # .docx files
    for filepath in glob_module.glob(os.path.join(folder_path, "*.docx")):
        doc = docx.Document(filepath)
        content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        docs.append({"content": content, "source": filepath})

    return docs

documents = load_all_files("data/policies")
print(f"Total documents loaded: {len(documents)}")

def simple_chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = []
for doc in documents:
    chunks.extend(simple_chunk_text(doc["content"]))

print(f"Total chunks created: {len(chunks)}")

import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="chroma_db")
ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="policy_docs",
    embedding_function=ef
)

ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, ids=ids)

print(f"Stored {len(chunks)} chunks in ChromaDB — Setup complete!")