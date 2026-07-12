import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

chroma_client = chromadb.PersistentClient(path="chroma_db")
ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="policy_docs",
    embedding_function=ef
)

load_dotenv()
groq_client = Groq()

def ask_policy_bot(question, model_name="llama-3.1-8b-instant"):
    search_results = collection.query(query_texts=[question], n_results=10)
    relevant_chunks = search_results["documents"][0]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful HR assistant. Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don't have that information in the policy documents."

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

print("PolicyBot ready! Type your question (or 'exit' to quit)\n")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    answer = ask_policy_bot(question)
    print(f"\nPolicyBot: {answer}\n")