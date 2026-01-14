import streamlit as st
from rag import process_inputs, generate_answer, clear_database

# --- Page Config ---
st.set_page_config(
    page_title="Real Estate AI Agent",
    page_icon="🏡",
    layout="wide"
)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🏗️ Agent Settings")

    # 1. Model Selector
    model_choice = st.selectbox(
        "AI Analyst Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Select the 'Brain' behind your agent."
    )

    # 2. Answer Style Selector
    answer_style = st.selectbox(
        "Analysis Persona",
        ["Investor", "Homebuyer", "Legal Expert"],
        index=0,
        help="Choose how you want the AI to analyze the data."
    )

    st.divider()

    st.subheader("📁 Property Data Source")

    # 3. URL Inputs
    url1 = st.text_input("Listing URL 1", placeholder="https://zillow.com/...")
    url2 = st.text_input("Listing URL 2", placeholder="https://redfin.com/...")

    # 4. PDF Uploader
    uploaded_pdfs = st.file_uploader(
        "Upload Contracts / Brochures",
        type="pdf",
        accept_multiple_files=True
    )

    # 5. Process Button
    if st.button("Analyze Property Data 🔍", type="primary"):
        urls = [u for u in [url1, url2] if u.strip()]

        if not urls and not uploaded_pdfs:
            st.error("Please provide a listing URL or property document.")
        else:
            with st.status("Reading property files...", expanded=True) as status:
                generator = process_inputs(urls=urls, pdf_files=uploaded_pdfs)
                for msg in generator:
                    st.write(msg)
                status.update(label="Analysis Complete! Ready to chat.", state="complete", expanded=False)

    st.divider()

    # --- NEW FEATURE: Mortgage Calculator ---
    with st.expander("💰 Mortgage Calculator", expanded=False):
        st.caption("Estimate your monthly costs")
        home_price = st.number_input("Home Price ($)", value=500000, step=10000)
        down_payment = st.number_input("Down Payment ($)", value=100000, step=5000)
        interest_rate = st.slider("Interest Rate (%)", 1.0, 10.0, 6.5, step=0.1)
        years = st.selectbox("Loan Term (Years)", [15, 30], index=1)

        # Calculation Logic
        loan_amount = home_price - down_payment
        monthly_rate = (interest_rate / 100) / 12
        num_payments = years * 12

        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / (
                        (1 + monthly_rate) ** num_payments - 1)
        else:
            monthly_payment = loan_amount / num_payments

        st.markdown(f"### **${monthly_payment:,.2f} / month**")
        st.caption(f"*Principal & Interest only. Loan Amount: ${loan_amount:,}*")

    st.divider()

    # --- NEW FEATURE: Download Report ---
    # Helper to format chat history into a text report
    chat_log = ["REAL ESTATE ANALYSIS REPORT\n" + "=" * 40 + "\n"]
    for msg in st.session_state.messages:
        role = msg['role'].upper()
        content = msg['content']
        chat_log.append(f"[{role}]:\n{content}\n")
        if 'sources' in msg:
            chat_log.append(f"SOURCES:\n{msg['sources']}\n")
        chat_log.append("-" * 40)

    chat_str = "\n".join(chat_log)

    st.download_button(
        label="📥 Download Analysis Report",
        data=chat_str,
        file_name="property_analysis_report.txt",
        mime="text/plain",
        help="Save the full conversation and sources as a text file."
    )

    # 6. Clear Memory
    if st.button("Clear Current Property 🗑️"):
        msg = clear_database()
        st.session_state.messages = []
        st.success(msg)

# --- Main Chat Interface ---
st.title("🏡 Real Estate AI Agent")
st.caption(f"**Expertise:** {answer_style} Mode | **Model:** {model_choice}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("📊 View Source Data & Context"):
                st.markdown(message["sources"])

# User Input
if prompt := st.chat_input("Ask about price, square footage, zoning, or investment potential..."):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing property details as a {answer_style}..."):
            try:
                answer, context_docs = generate_answer(prompt, model_choice, answer_style)

                st.markdown(answer)

                # Format Sources
                source_text = ""
                for i, doc in enumerate(context_docs):
                    src = doc.metadata.get("source", "Unknown")
                    content_preview = doc.page_content[:200].replace("\n", " ") + "..."
                    source_text += f"**Source {i + 1} ({src}):**\n>{content_preview}\n\n"

                with st.expander("📊 View Source Data & Context"):
                    st.markdown(source_text)

                # 3. Save to History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_text
                })

            except RuntimeError:
                st.error("⚠️ No property loaded. Please add a URL or PDF in the sidebar first.")
            except Exception as e:
                st.error(f"An error occurred: {e}")