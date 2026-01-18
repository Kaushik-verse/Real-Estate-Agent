import streamlit as st
from rag import process_inputs, generate_answer, clear_database

st.set_page_config(
    page_title="Real Estate AI Agent",
    page_icon="🏡",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🏗️ Agent Settings")

    model_choice = st.selectbox(
        "AI Analyst Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )

    answer_style = st.selectbox(
        "Analysis Persona",
        ["Investor", "Homebuyer", "Legal Expert"],
        index=0
    )

    st.divider()
    st.subheader("📁 Property Data Source")

    url1 = st.text_input("Listing URL 1")
    url2 = st.text_input("Listing URL 2")

    uploaded_pdfs = st.file_uploader(
        "Upload Contracts / Brochures",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Analyze Property Data 🔍", type="primary"):
        urls = [u for u in [url1, url2] if u.strip()]

        if not urls and not uploaded_pdfs:
            st.error("Please provide a URL or PDF.")
        else:
            with st.status("Processing property data...", expanded=True):
                for msg in process_inputs(urls, uploaded_pdfs):
                    st.write(msg)

    st.divider()

    with st.expander("💰 Mortgage Calculator"):
        price = st.number_input("Home Price", 500000)
        down = st.number_input("Down Payment", 100000)
        rate = st.slider("Interest Rate (%)", 1.0, 10.0, 6.5)
        years = st.selectbox("Loan Term", [15, 30], index=1)

        loan = price - down
        r = rate / 100 / 12
        n = years * 12

        payment = loan * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        st.markdown(f"### ${payment:,.2f} / month")

    st.divider()

    chat_log = ["REAL ESTATE ANALYSIS REPORT\n" + "=" * 40]
    for m in st.session_state.messages:
        chat_log.append(f"\n[{m['role'].upper()}]\n{m['content']}")
        if "sources" in m:
            chat_log.append(f"\nSOURCES:\n{m['sources']}")

    st.download_button(
        "📥 Download Report",
        "\n".join(chat_log),
        "property_analysis_report.txt",
        "text/plain"
    )

    if st.button("Clear Current Property 🗑️"):
        st.session_state.messages = []
        st.success(clear_database())

st.title("🏡 Real Estate AI Agent")
st.caption(f"Persona: {answer_style} | Model: {model_choice}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "sources" in m:
            with st.expander("📊 Sources"):
                st.markdown(m["sources"])

if prompt := st.chat_input("Ask about price, ROI, zoning, or location..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing property data..."):
            answer, docs = generate_answer(prompt, model_choice, answer_style)
            st.markdown(answer)

            sources = ""
            for i, d in enumerate(docs):
                src = d.metadata.get("source", "Unknown")
                preview = d.page_content[:200].replace("\n", " ") + "..."
                sources += f"**Source {i+1} ({src})**\n>{preview}\n\n"

            with st.expander("📊 Sources"):
                st.markdown(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
