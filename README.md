# 🏦 Audit Intelligence System (AIS)

[cite_start]An autonomous, local-first Audit Intelligence System designed to analyze, label, and summarize financial documents, primarily bank statements, using a sophisticated Agentic framework (CrewAI) and persistent local storage (MongoDB)[cite: 2, 5, 7].

---

## ✨ Features

[cite_start]This system is built around core requirements to create a lightweight, fully autonomous auditing assistant[cite: 7, 8]:

* [cite_start]**Agentic Orchestration:** Uses **CrewAI** to coordinate specialized agents (Planner, Executor, Labeler, Reviewer)[cite: 3, 9, 24].
* [cite_start]**Hybrid RAG:** Performs Retrieval-Augmented Generation (RAG) using FAISS or cosine similarity[cite: 3, 28].
* [cite_start]**Auto-Labeling:** Automatically labels bank statement fields (DATE, DEBIT, CREDIT, CATEGORY) using a hybrid **Rule-Based + LLM** method[cite: 3, 16, 30].
* [cite_start]**Local Persistence:** Stores all documents, labels, queries, logs, and model metrics in a local **MongoDB** database[cite: 5, 20, 41, 42].
* [cite_start]**Auto Fine-Tuning Loop:** Detects new labeled or Q&A data and automatically triggers **LoRA/PEFT** fine-tuning on a lightweight local LLM[cite: 4, 18, 47].
* [cite_start]**Reporting & Analysis:** Generates data analysis reports using **pandas** and **seaborn** visualizations, exported via `python-docx`[cite: 31, 32, 33].
* [cite_start]**Offline Functionality:** The entire system runs fully locally and offline[cite: 5].

---

## 🧠 Technical Deep Dive: How the Model Works

The Audit Intelligence System operates on a sequential, collaborative agent workflow with persistent memory.

### 1. Agentic Orchestration and Flow

[cite_start]The system uses **CrewAI** (based on LangChain/LangGraph principles) to manage the workflow in a structured, sequential process[cite: 3, 24].

* [cite_start]**Flow:** The main operational flow follows a Planner → Executor → Critic → Labeling sequence, managed by task dependencies and shared context memory[cite: 25].
* [cite_start]**API Design:** Each agent exposes a clear callable API (functions or classes) with structured input/output, primarily utilizing JSON for inter-agent communication.

| Agent | Role & Goal | Key Tools/Functions |
| :--- | :--- | :--- |
| **Planner Agent** | [cite_start]Breaks down user queries into sequential subtasks[cite: 11]. | Query Breakdown |
| **Labeling Agent** | [cite_start]Automatically labels bank statement transactions[cite: 16]. | [cite_start]Hybrid Labeling (Rule-Based + LLM) [cite: 36] |
| **Executor Agent** | [cite_start]Handles document parsing, RAG, data analysis, and summary generation[cite: 14, 31]. | [cite_start]RAG Retrieval, Pandas Analysis, Report Generation [cite: 33] |
| **Reviewer Agent** | [cite_start]Validates responses, checks factual consistency, and requests retries[cite: 15]. | Confidence Check, Factual Validation |

### 2. Hybrid RAG Implementation

[cite_start]The system employs a Hybrid RAG approach:

1.  **Parsing:** Financial documents (e.g., PDF bank statements) are parsed into structured chunks.
2.  **Indexing:** Document chunks are converted into embedding vectors using a Sentence Transformer model.
3.  [cite_start]**Storage:** The vectors are indexed and stored using **FAISS** for rapid similarity search.
4.  [cite_start]**Retrieval:** User queries are matched against the FAISS index (vector similarity) and simultaneously against the MongoDB structured data (keyword/semantic search).
5.  [cite_start]**Generation:** The retrieved context (text chunks + structured data) is passed to the local LLM for accurate, grounded answer generation[cite: 3].

### 3. The Auto-Labeling Mechanism

[cite_start]The **Labeling Agent** uses a robust two-step process for categorization[cite: 30]:

1.  [cite_start]**Rule-Based Matching:** Initial classification is attempted using pre-defined rules (e.g., matching keywords like "SALARY," "RENT," "UTILITY")[cite: 16]. This is fast and reliable for common transactions.
2.  [cite_start]**LLM Fallback:** If the rule-based system yields "UNCATEGORIZED," a lightweight local LLM is used to interpret the transaction description and assign a category[cite: 36].
3.  [cite_start]**Persistence:** The final labeled output (containing fields like DATE, DESCRIPTION, DEBIT, CREDIT, BALANCE, CATEGORY) is stored as JSON/CSV locally and persisted in the MongoDB `labeled_data` collection[cite: 35, 38].

### 4. Local Fine-Tuning Loop (LoRA/PEFT)

[cite_start]To ensure the local LLM is specialized for audit tasks, an automatic fine-tuning loop is implemented[cite: 4, 18]:

1.  [cite_start]**Data Detection:** The system continuously monitors the database and local directories (`datasets/labeled_data/` and `datasets/qna_data/`) for new entries[cite: 18].
2.  [cite_start]**Trigger:** When new data is detected, the `AutoFineTuner` component is triggered[cite: 39].
3.  [cite_start]**Fine-Tuning:** It utilizes **LoRA/PEFT** (Parameter-Efficient Fine-Tuning) to train only a small number of parameters (adapters) on the base LLM[cite: 4, 47]. This saves significant computational resources compared to full fine-tuning.
4.  [cite_start]**Logging:** Key metrics (loss, accuracy, training time) are logged to the MongoDB `model_metrics` collection for traceability and evaluation[cite: 50, 44].

---

## ⚙️ Technical Requirements & Stack

| Component | Technology | Requirement Reference |
| :--- | :--- | :--- |
| **Agent Orchestration** | [cite_start]CrewAI (or LangGraph) [cite: 3, 24] | |
| **Database** | [cite_start]MongoDB [cite: 5, 41] | |
| **Retrieval** | [cite_start]FAISS / Cosine Similarity  | |
| **Fine-Tuning** | [cite_start]LoRA / PEFT [cite: 4, 47] | |
| **UI/Interface** | [cite_start]Streamlit  | |
| **Data Analysis** | [cite_start]Pandas, Seaborn [cite: 31, 32] | |
| **Reporting** | [cite_start]Python-docx / ReportLab [cite: 33] | |
| **Local LLM** | [cite_start]Lightweight local models [cite: 48, 36] | |

---

## 📂 Project Structure
