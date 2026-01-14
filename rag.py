# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.
# Refactored for Real Estate AI Agent

import os
import ssl
import nltk
import tempfile
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

# --- SSL & NLTK Fixes ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# --- Imports ---
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Dynamic Import for Chains
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# --- Configuration ---
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate_agent_db"  # Updated DB Name

# Globals
vector_store = None
current_llm_model = None
llm = None


def get_vector_store():
    global vector_store
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )
    return vector_store


def get_llm(model_name="llama-3.3-70b-versatile"):
    global llm, current_llm_model
    if llm is None or model_name != current_llm_model:
        llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            max_tokens=1000
        )
        current_llm_model = model_name
    return llm


def clear_database():
    vs = get_vector_store()
    vs.reset_collection()
    return "Property data and market analysis cleared! 🧹"


def process_inputs(urls=None, pdf_files=None):
    yield "Initializing Real Estate Agent..."
    vs = get_vector_store()
    documents = []

    # 1. Process URLs
    if urls:
        yield f"Analyzing content from {len(urls)} listings/reports... 🌐"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        loader = UnstructuredURLLoader(urls=urls, headers=headers)
        documents.extend(loader.load())

    # 2. Process PDFs
    if pdf_files:
        yield f"Processing {len(pdf_files)} property documents... 📄"
        for uploaded_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(docs)
            finally:
                os.remove(tmp_path)

    if not documents:
        yield "No valid property data found."
        return

    # 3. Split and Store
    yield "Indexing property details & market data... 🏠"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )
    splitted_docs = text_splitter.split_documents(documents)

    yield f"Storing {len(splitted_docs)} data points in knowledge base... 🧠"
    uuids = [str(uuid4()) for _ in range(len(splitted_docs))]
    vs.add_documents(splitted_docs, ids=uuids)

    yield "Success! Real Estate Agent is ready to answer. ✅"


def generate_answer(query, model_name, answer_style="Investor"):
    vs = get_vector_store()
    chat_llm = get_llm(model_name)
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    # --- CUSTOMIZED REAL ESTATE PROMPTS ---
    style_instruction = ""
    if answer_style == "Investor":
        style_instruction = "Focus on ROI, cap rates, growth potential, and risks. Be data-driven."
    elif answer_style == "Homebuyer":
        style_instruction = "Focus on lifestyle, amenities, school districts, and comfort. Be friendly."
    elif answer_style == "Legal Expert":
        style_instruction = "Focus on zoning laws, contract clauses, and compliance. Be formal and precise."

    system_prompt = (
        "You are an expert Real Estate AI Agent. "
        "Your goal is to assist users in analyzing property listings, market trends, and investment documents. "
        "Use the provided context to answer the question accurately. "
        "If the answer is not in the context, strictly state that you don't have that information. "
        f"\n\n**Tone Instruction:** {style_instruction}"
        "\n\n**Context Data:**\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chat_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    return response["answer"], response["context"]