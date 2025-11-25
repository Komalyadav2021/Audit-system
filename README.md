# 🏦 Local Audit Intelligence System

An autonomous, local-first Audit Intelligence System designed to analyze, label, and summarize financial documents, primarily bank statements, using a sophisticated Agentic framework (CrewAI) and persistent local storage (MongoDB).

---

## ✨ Features

This system is built around core requirements to create a lightweight, fully autonomous auditing assistant:

* [cite_start]**Agentic Orchestration:** Uses **CrewAI** to coordinate specialized agents (Planner, Executor, Labeler, Reviewer)[cite: 3, 9, 23, 24].
* [cite_start]**Hybrid RAG:** Performs Retrieval-Augmented Generation (RAG) using FAISS or cosine similarity[cite: 3, 28].
* [cite_start]**Auto-Labeling:** Automatically labels bank statement fields (DATE, DEBIT, CREDIT, CATEGORY) using a hybrid **Rule-Based + LLM** method[cite: 3, 16, 30, 34, 35, 36].
* [cite_start]**Local Persistence:** Stores all documents, labels, queries, logs, and model metrics in a local **MongoDB** database[cite: 5, 20, 21, 41, 42].
* [cite_start]**Auto Fine-Tuning Loop:** Detects new labeled or Q&A data and automatically triggers **LoRA/PEFT** fine-tuning on a lightweight local LLM[cite: 4, 18, 47, 48].
* [cite_start]**Data Generation:** Auto-generates Q&A pairs from parsed documents for training data enrichment[cite: 4, 18, 46].
* [cite_start]**Reporting:** Generates data analysis reports using **pandas** and **seaborn** visualizations, exported via `python-docx`[cite: 31, 32, 33].
* [cite_start]**Offline Functionality:** The entire system runs fully locally and offline[cite: 5].

---

## ⚙️ Technical Requirements & Stack

| Component | Technology | Requirement Reference |
| :--- | :--- | :--- |
| **Agent Orchestration** | [cite_start]CrewAI / Python |  |
| **Database** | [cite_start]MongoDB | [cite: 5, 41] |
| **Retrieval** | [cite_start]FAISS / Cosine Similarity | [cite: 28] |
| **Fine-Tuning** | [cite_start]LoRA / PEFT (Hugging Face ecosystem) |  |
| **UI/Interface** | [cite_start]Streamlit |  |
| **Data Analysis** | [cite_start]Pandas, Seaborn | [cite: 31, 32] |
| **Local LLM** | [cite_start]Lightweight model (e.g., Llama 2 via Ollama) | [cite: 36, 48] |

---

## 📂 Project Structure
