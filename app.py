
import streamlit as st
import faiss
import numpy as np
import os
import io

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 AI-Based PDF Question Answering Chatbot")
st.write("Upload a PDF and ask questions about its content.")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Connect to Groq
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# Process uploaded PDF only when its contents change
@st.cache_resource
def process_pdf(pdf_bytes):

    reader = PdfReader(io.BytesIO(pdf_bytes))

    chunks = []
    chunk_size = 500
    overlap = 50

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    if not chunks:
        return None, []

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks]
    )

    vectors = np.array(embeddings, dtype=np.float32)

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, chunks


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Please upload a PDF to start.")
    st.stop()

# Process the uploaded PDF
index, chunks = process_pdf(uploaded_file.getvalue())

if index is None:
    st.error("No readable text found in this PDF.")
    st.stop()

st.success("PDF processed successfully!")

st.write("Total number of chunks:", len(chunks))

# Retrieve relevant chunks
def retrieve_chunks(question, k=3):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(k, index.ntotal)
    )

    retrieved = []

    for chunk_id in indices[0]:

        if chunk_id != -1:
            retrieved.append(chunks[chunk_id])

    return retrieved


# Generate answer using Groq
def generate_answer(question):

    retrieved = retrieve_chunks(question)

    context = ""

    for item in retrieved:
        context += (
            f"[Page {item['page']}]\n"
            f"{item['text']}\n\n"
        )

    prompt = f"""
    You are a PDF question-answering assistant.

    Answer using only the supplied PDF context.

    If the answer is not available, say:
    The answer is not available in the PDF.

    Mention the relevant page numbers when possible.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )

    answer = response.choices[0].message.content

    source_pages = sorted(
        set(item["page"] for item in retrieved)
    )

    return answer, source_pages, retrieved


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "pages" in message:

            with st.expander("View source pages"):
                st.write("Relevant PDF pages:", message["pages"])

                for source in message["sources"]:
                    st.write(
                        f"Page {source['page']}:"
                    )
                    st.write(source["text"])


# Ask a question
question = st.chat_input("Ask a question about your PDF...")

if question:

    # Display and save user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching PDF and generating answer..."):

            try:
                answer, pages, sources = generate_answer(question)

                st.markdown(answer)

                with st.expander("View source pages"):
                    st.write("Relevant PDF pages:", pages)

                    for source in sources:
                        st.write(f"Page {source['page']}:")
                        st.write(source["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "pages": pages,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
