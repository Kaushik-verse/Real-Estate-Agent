import os
import tempfile
from uuid import uuid4
import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "/tmp/chroma"
COLLECTION_NAME = "real_estate_agent_db"

vector_store = None
llm = None
current_model = None


def get_vector_store():
    global vector_store
    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
    return vector_store


def get_llm(model):
    global llm, current_model
    if llm is None or model != current_model:
        llm = ChatGroq(model=model, temperature=0.7, max_tokens=1000)
        current_model = model
    return llm


def clear_database():
    get_vector_store().reset_collection()
    return "Property data cleared."


def process_inputs(urls=None, pdf_files=None):
    vs = get_vector_store()
    documents = []

    if urls:
        yield "Reading property URLs..."
        loader = UnstructuredURLLoader(urls=urls)
        documents.extend(loader.load())

    if pdf_files:
        yield "Processing PDFs..."
        for f in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                path = tmp.name
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = f.name
            documents.extend(docs)
            os.remove(path)

    if not documents:
        yield "No data found."
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    ids = [str(uuid4()) for _ in chunks]
    vs.add_documents(chunks, ids=ids)

    yield "Property data indexed successfully."


def generate_answer(query, model, persona):
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm = get_llm(model)

    tone = {
        "Investor": "Focus on ROI, risks, appreciation, and numbers.",
        "Homebuyer": "Focus on comfort, amenities, and lifestyle.",
        "Legal Expert": "Focus on zoning, contracts, and compliance."
    }[persona]

    system_prompt = (
        "You are a Real Estate AI Agent.\n"
        f"Tone: {tone}\n"
        "Use only the provided context.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    result = rag_chain.invoke({"input": query})
    return result["answer"], result["context"]
