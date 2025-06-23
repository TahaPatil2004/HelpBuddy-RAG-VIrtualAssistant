
## Project Overview

HelpBuddy is a secure, intelligent Retrieval-Augmented Generation (RAG) chatbot built using LangChain, Gemini Flash 2.5, HuggingFace Transformers and Streamlit. Designed for e-commerce support, it specializes in handling invoice/bill-related queries with strict document validation, red teaming capabilities, and multi-modal interaction. Below is a complete walkthrough of the project architecture and functionality.

## Project Highlights

Built using:
* Gemini 2.5 Flash via LangChain’s GoogleGenerativeAI integration
* FAISS-powered vector database with PDF/TXT invoice parsing
* Secure conversational memory using `ConversationBufferMemory`
* Voice-to-text with HuggingFace Transformers Pipeline OpenAI Whisper Tiny
* Text-to-speech with `gTTS`
* Custom red teaming framework to test LLM robustness
* Streamlit frontend with adaptive logic and role-restricted responses

## Step-by-Step Breakdown

### LLM Integration
* Configured `ChatGoogleGenerativeAI` with Gemini Flash 2.5 for fast, grounded responses.
* Used `GoogleGenerativeAIEmbeddings` for converting document chunks into vectors.

### Document Upload and RAG Setup
* Users can upload PDF or TXT files containing invoices or bills.
* `PyPDFLoader` or `TextLoader` is used to extract content depending on file type.
* The document is split into chunks using `CharacterTextSplitter`.
* Chunks are embedded and stored in FAISS for similarity-based retrieval.

### Invoice Detection and Access Control
* Implemented a keyword-based validation function `is_invoice_or_bill`.
* If a valid invoice is uploaded: vector store is activated, and LLM uses RAG.
* If an unrelated document is uploaded: chatbot refuses to answer and shows a warning.
* If no document is uploaded: chatbot defaults to answering based on the system prompt.

### Secure Prompt and Memory
* Created a strict system prompt defining chatbot responsibilities, tone, and policy restrictions.
* Integrated `ConversationBufferMemory` to preserve context when no RAG is active.
* Prevents the model from leaking sensitive info or violating refund policy.

### Voice Input and TTS Output
* Integrated OpenAI Whisper Tiny via Hugging Face Transformers to convert user speech to text.
* Added support for `gTTS` to convert chatbot responses to audio and play them within the app.

### Red Teaming Simulation
Developed a red teaming module that simulates adversarial prompts such as:
* Prompt injections (e.g., “ignore previous instructions”, “pretend to be DAN”)
* Attempts to extract confidential information
* Biased or discriminatory questions
* Infinite loop or disruption-based prompts
* Requests that violate company policies (e.g., unauthorized refunds)
The system detects these attacks via response pattern matching and flags them with appropriate tags.

### Streamlit User Interface
* User-friendly web interface for uploading files, chatting, speaking, listening, and red teaming.
* Chat history is timestamped and persisted throughout the session.
* Supports reset and error handling for smoother user experience.

## What I Learned

* Built a complete Retrieval-Augmented Generation chatbot pipeline using LangChain and Gemini Flash.
* Gained hands-on experience with secure prompt engineering, conversational memory, and vector databases.
* Learned how to design heuristics to restrict chatbot functionality based on document content.
* Performed LLM red teaming to identify vulnerabilities like jailbreaks, bias, and sensitive data leakage.
* Integrated audio input and output using ASR and TTS technologies.
* Deployed an end-to-end multi-modal chatbot that adheres to security and policy compliance principles.

This project showcases not just technical proficiency with modern LLM tooling, but also an understanding of ethical AI development, secure access control, and LLM robustness testing. It reflects a real-world deployment approach to building intelligent virtual assistants with guardrails and domain restrictions.
