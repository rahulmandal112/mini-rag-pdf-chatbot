import streamlit as st
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# CONFIG
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# HELPER FUNCTIONS


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings


def retrieve(query, chunks, index, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks


def generate_answer(context, question, chat_history):
    history_text = ""
    for item in chat_history[-3:]:  # last 3 interactions
        history_text += f"User: {item['question']}\nAssistant: {item['answer']}\n"

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided document context to answer.
If the answer is not found in the context, say:
"I cannot find this information in the document."

Previous conversation:
{history_text}

Document Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content



# STREAMLIT UI


st.set_page_config(page_title="Mini RAG PDF Bot", layout="wide")

st.title("📄 Mini RAG Bot – PDF Q&A ")
st.write("Upload a PDF and ask questions grounded in the document.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("Could not extract text from this PDF.")
    else:
        chunks = chunk_text(text)
        index, _ = create_faiss_index(chunks)

        st.success("PDF processed successfully!")

        question = st.text_input("Ask a question about the document:")

        if question:
            retrieved_chunks = retrieve(question, chunks, index, k=3)
            context = "\n\n".join(retrieved_chunks)

            answer = generate_answer(
                context,
                question,
                st.session_state.chat_history
            )

            # Save to memory
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })

            st.subheader("Answer")
            st.write(answer)

            with st.expander("🔍 Retrieved Context Chunks"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(chunk)
                    st.write("---")

            with st.expander("💬 Conversation History"):
                for i, chat in enumerate(st.session_state.chat_history):
                    st.markdown(f"**Q{i+1}:** {chat['question']}")
                    st.markdown(f"**A{i+1}:** {chat['answer']}")
                    st.write("---")