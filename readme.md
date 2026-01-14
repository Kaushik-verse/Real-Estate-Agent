# 🏡 Real Estate AI Agent

An intelligent research assistant designed for real estate investors, homebuyers, and agents. This tool uses **Retrieval-Augmented Generation (RAG)** to analyze property listings (URLs) and documents (PDFs), providing data-driven insights with customizable expert personas.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Groq](https://img.shields.io/badge/AI-Groq%20(Llama3)-orange)

## 🚀 Key Features

* **🎭 Multi-Persona Analysis:**
    * **Investor Mode:** Focuses on ROI, Cap Rates, cash flow, and risk assessment.
    * **Homebuyer Mode:** Focuses on lifestyle, amenities, school districts, and livability.
    * **Legal Expert Mode:** Focuses on zoning laws, compliance, and contract clauses.
* **📂 Hybrid Data Processing:** Seamlessly combines public data (Listing URLs from Zillow/Redfin) with private documents (Inspection Reports, Contracts in PDF).
* **🧠 Conversation Memory:** Remembers context from previous questions for a natural chat experience.
* **🔍 Source Transparency:** "Glass-box" AI that cites exact sources and displays the raw text chunks used to generate every answer.
* **🤖 Dynamic Model Switching:** Choose between **Llama-3.3-70b** (High Intelligence) and **Mixtral-8x7b** (Speed).

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain (v0.3/v1.0 Compatible)
* **LLM Provider:** Groq (Llama 3 & Mixtral)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Parsers:** Unstructured (for URLs), PyPDF (for PDFs)

## ⚙️ Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/yourusername/real-estate-ai-agent.git](https://github.com/yourusername/real-estate-ai-agent.git)
cd real-estate-ai-agent

```

**2. Create a Virtual Environment**

```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate

```

**3. Install Dependencies**

```bash
pip install -r requirements.txt

```

**4. Configure API Keys**
Create a `.env` file in the root directory and add your free Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_api_key_here

```

**5. Run the Application**

```bash
streamlit run main.py

```

---

## 🧪 Example Test Cases

Use these scenarios to verify the AI's "Persona" and "Hybrid Search" capabilities.

### 🏠 Scenario 1: The "Homebuyer" vs. "Investor" Test

*Goal: See how the AI changes its answer based on the selected persona.*

**Setup:**

1. **URL:** Input a Wikipedia link for a city or a specific building (e.g., `https://en.wikipedia.org/wiki/Empire_State_Building`).
2. **Process:** Click "Analyze Property Data".

**Test A (Homebuyer Mode):**

* **Select Persona:** "Homebuyer"
* **Question:** *"Tell me about this property."*
* **Expected Output:** The AI describes the views, the history, the design, and the location's prestige. It feels emotional and descriptive.

**Test B (Investor Mode):**

* **Select Persona:** "Investor"
* **Question:** *"Tell me about this property."*
* **Expected Output:** The AI focuses on the commercial office space, tenant occupancy, renovation costs, and its value as a commercial asset. It feels analytical and dry.

### 🕵️ Scenario 2: The "Hidden Flaw" Hybrid Test (URL + PDF)

*Goal: Prove the AI can find "secret" info in a PDF that isn't on the public website.*

**Setup:**

1. **Create a Dummy PDF:** Create a text file named `inspection_report.pdf` with the text:
> "CONFIDENTIAL INSPECTION: The foundation has severe water damage requiring $50,000 in repairs."


2. **URL:** Input a generic real estate listing (or a Wikipedia page about a building).
3. **Upload:** Upload the `inspection_report.pdf`.
4. **Process:** Click "Analyze Property Data".

**Test:**

* **Question:** *"Are there any risks with this property?"*
* **Expected Output:** The AI should mention the public info from the URL **AND** warn you about the "severe water damage" found in the PDF.
* **Verification:** Click **"📊 View Source Data"** to see it citing your PDF.

---

## ⚠️ Troubleshooting

* **"SSL Certificate Verify Failed":**
* This is a common macOS issue. The code includes an auto-fix, but if it fails, run: `/Applications/Python 3.x/Install Certificates.command`


* **"Access Denied" on URLs:**
* Some sites (like Bloomberg or WSJ) have strict firewalls. The agent uses a browser User-Agent to bypass basic blocks, but Wikipedia or smaller news sites work best for testing.



## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features (e.g., Mortgage Calculator integration, Map View).

## 📄 License

This project is licensed under the MIT License.

```

```