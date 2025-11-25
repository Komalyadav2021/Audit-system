# 🏦  Audit Intelligence System (AIS)

An autonomous, local-first Audit Intelligence System designed to analyze, label, and summarize financial documents, primarily bank statements, using a sophisticated Agentic framework (CrewAI) and persistent local storage (MongoDB).

---

## ✨ Features

This system is built around core requirements to create a lightweight, fully autonomous auditing assistant:

* **Agentic Orchestration:** Uses **CrewAI** to coordinate specialized agents (Planner, Executor, Labeler, Reviewer).
* **Hybrid RAG:** Performs Retrieval-Augmented Generation (RAG) using FAISS or cosine similarity.
* **Auto-Labeling:** Automatically labels bank statement fields (DATE, DEBIT, CREDIT, CATEGORY) using a hybrid **Rule-Based + LLM** method.
* **Local Persistence:** Stores all documents, labels, queries, logs, and model metrics in a local **MongoDB** database.
* **Auto Fine-Tuning Loop:** Detects new labeled or Q&A data and automatically triggers **LoRA/PEFT** fine-tuning on a lightweight local LLM.
* **Reporting & Analysis:** Generates data analysis reports using **pandas** and **seaborn** visualizations, exported via `python-docx`.
* **Offline Functionality:** The entire system runs fully locally and offline.

---

## 🧠 Technical Deep Dive: How the Model Works

The Audit Intelligence System operates on a sequential, collaborative agent workflow with persistent memory. 

### 1. Agentic Orchestration and Flow

The system uses **CrewAI** (based on LangChain/LangGraph principles) to manage the workflow in a structured, sequential process.

* **Flow:** The main operational flow follows a **Planner → Executor → Critic → Labeling** sequence, managed by task dependencies and shared context memory.
* **API Design:** Each agent exposes a clear callable API (functions or classes) with structured input/output, primarily utilizing JSON for inter-agent communication.

| Agent | Role & Goal | Key Tools/Functions |
| :--- | :--- | :--- |
| **Planner Agent** | Breaks down user queries into sequential subtasks. | Query Breakdown |
| **Labeling Agent** | Automatically labels bank statement transactions. | Hybrid Labeling (Rule-Based + LLM) |
| **Executor Agent** | Handles document parsing, RAG, data analysis, and summary generation. | RAG Retrieval, Pandas Analysis, Report Generation |
| **Reviewer Agent** | Validates responses, checks factual consistency, and requests retries. | Confidence Check, Factual Validation |

---

### 2. Hybrid RAG Implementation

The system employs a Hybrid RAG approach:

1.  **Parsing:** Financial documents (e.g., PDF bank statements) are parsed into structured chunks.
2.  **Indexing:** Document chunks are converted into embedding vectors using a Sentence Transformer model.
3.  **Storage:** The vectors are indexed and stored using **FAISS** for rapid similarity search.
4.  **Retrieval:** User queries are matched against the FAISS index (vector similarity) and simultaneously against the MongoDB structured data (keyword/semantic search).
5.  **Generation:** The retrieved context (text chunks + structured data) is passed to the local LLM for accurate, grounded answer generation. 

---

### 3. The Auto-Labeling Mechanism

The **Labeling Agent** uses a robust two-step process for categorization:

1.  **Rule-Based Matching:** Initial classification is attempted using pre-defined rules (e.g., matching keywords like "SALARY," "RENT," "UTILITY"). This is fast and reliable for common transactions.
2.  **LLM Fallback:** If the rule-based system yields "UNCATEGORIZED," a lightweight local LLM is used to interpret the transaction description and assign a category.
3.  **Persistence:** The final labeled output (containing fields like **DATE, DESCRIPTION, DEBIT, CREDIT, BALANCE, CATEGORY**) is stored as JSON/CSV locally and persisted in the MongoDB `labeled_data` collection.

---

### 4. Local Fine-Tuning Loop (LoRA/PEFT)

To ensure the local LLM is specialized for audit tasks, an automatic fine-tuning loop is implemented:

1.  **Data Detection:** The system continuously monitors the database and local directories (`datasets/labeled_data/` and `datasets/qna_data/`) for new entries.
2.  **Trigger:** When new data is detected, the `AutoFineTuner` component is triggered.
3.  **Fine-Tuning:** It utilizes **LoRA/PEFT** (Parameter-Efficient Fine-Tuning) to train only a small number of parameters (adapters) on the base LLM. This saves significant computational resources compared to full fine-tuning. 
4.  **Logging:** Key metrics (loss, accuracy, training time) are logged to the MongoDB `model_metrics` collection for traceability and evaluation.

---

## ⚙️ Technical Requirements & Stack

| Component | Technology |
| :--- | :--- |
| **Agent Orchestration** | CrewAI (or LangGraph) |
| **Database** | MongoDB |
| **Retrieval** | FAISS / Cosine Similarity |
| **Fine-Tuning** | LoRA / PEFT |
| **UI/Interface** | Streamlit |
| **Data Analysis** | Pandas, Seaborn |
| **Reporting** | Python-docx / ReportLab |
| **Local LLM** | Lightweight local models |

---
