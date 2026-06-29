# CollegeAiAssistance
Developed a Retrieval-Augmented Generation (RAG) based College Information Assistant using LangChain, FAISS, and Llama 3.1 to answer student queries from institutional documents.

<img width="1907" height="905" alt="image" src="https://github.com/user-attachments/assets/e49c18a0-284e-44a2-9fb5-5ce84bf9eff1" />

<img width="1913" height="902" alt="image" src="https://github.com/user-attachments/assets/305de8ab-8f17-4541-9d76-bbb905107385" />


# Theory

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Hugging Face**, and **FAISS** to build an intelligent college information assistant. Instead of relying solely on the Large Language Model (LLM), the system retrieves relevant information from college documents before generating a response, ensuring accurate and context-aware answers.

## Architecture

```
            User Question
                  │
                  ▼
          RunnablePassthrough
                  │
                  ▼
          FAISS Retriever (MMR)
                  │
                  ▼
        Relevant Document Chunks
                  │
                  ▼
        Convert Documents to Text
                  │
                  ▼
          Prompt Template
                  │
                  ▼
      Llama 3.1 8B (Hugging Face)
                  │
                  ▼
      Pydantic Output Parser
                  │
                  ▼
          Structured JSON Output
```

---

# Working

## 1. Loading Environment Variables

The project uses **python-dotenv** to securely load environment variables such as the Hugging Face API token.

```python
load_dotenv()
```

Keeping API keys outside the source code improves security and makes deployment easier.

---

## 2. Large Language Model (LLM)

The chatbot uses the **Meta Llama 3.1 8B Instruct** model hosted on Hugging Face.

```python
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
```

This model is responsible for generating natural language responses after receiving the retrieved context.

---

## 3. Embedding Model

Documents are converted into dense vector representations using:

```python
BAAI/bge-small-en-v1.5
```

Embeddings capture semantic meaning rather than exact keyword matching, allowing the chatbot to retrieve relevant information even when users phrase questions differently.

---

## 4. Document Loading

The application loads all `.txt` files from the `data` directory using the `DirectoryLoader`.

```python
loader = DirectoryLoader(
    path="data",
    glob="*.txt",
    loader_cls=TextLoader
)
```

Each text file represents a knowledge source for the chatbot.

---

## 5. Text Splitting

Large documents are divided into smaller overlapping chunks using a `RecursiveCharacterTextSplitter`.

Parameters:

- Chunk Size: **700 characters**
- Chunk Overlap: **150 characters**

Chunking improves retrieval accuracy while overlap preserves context between adjacent chunks.

Example:

```
Original Document

-------------------------
Paragraph A
Paragraph B
Paragraph C
-------------------------

↓

Chunk 1
Paragraph A
Paragraph B

Chunk 2
Paragraph B
Paragraph C
```

---

## 6. Vector Database (FAISS)

Each document chunk is converted into embeddings and stored inside a **FAISS** vector database.

```python
FAISS.from_documents(...)
```

FAISS enables efficient similarity search across thousands of document chunks.

---

## 7. Retriever

The vector database is converted into a retriever.

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10
    }
)
```

The project uses **Maximum Marginal Relevance (MMR)** retrieval.

### Why MMR?

Instead of returning four nearly identical chunks, MMR returns:

- Relevant chunks
- Diverse chunks
- Less redundant information

Parameters:

| Parameter | Meaning |
|-----------|---------|
| fetch_k=10 | Retrieve the top 10 candidate chunks |
| k=4 | Return the best 4 diverse chunks |

This improves answer quality by providing broader context.

---

## 8. Context Creation

Retrieved documents are combined into a single context string.

```python
documents_to_text()
```

Example:

```
Chunk 1

Admission Process...

Chunk 2

Required Documents...

Chunk 3

Eligibility Criteria...
```

↓

```
Complete Context
```

This context is supplied to the LLM.

---

## 9. Prompt Engineering

A custom prompt instructs the LLM to:

- Answer **only** from the retrieved context.
- Summarize information clearly.
- Avoid hallucinating.
- Return structured output.

If no relevant information exists, the model responds with:

```json
{
  "answer": "Information not found",
  "source_found": false
}
```

---

## 10. Structured Output

Instead of plain text, the chatbot returns a structured JSON object defined using Pydantic.

```python
class CollegeAnswer(BaseModel):
    answer: str
    source_found: bool
```

Example:

```json
{
    "answer": "The admission process begins with online registration followed by document verification.",
    "source_found": true
}
```

Structured responses make it easier to integrate the chatbot into web applications or APIs.

---

## 11. LangChain Runnable Pipeline

The entire workflow is implemented using LangChain's Runnable interface.

```
User Question
        │
        ▼
RunnableParallel
        │
 ┌──────┴────────┐
 │               │
 ▼               ▼
Retriever     User Question
 │               │
 ▼               ▼
Context     RunnablePassthrough
 └──────┬────────┘
        ▼
 Prompt Template
        ▼
 Llama 3.1 Model
        ▼
Pydantic Parser
        ▼
 Structured Output
```

This modular pipeline improves readability, maintainability, and scalability.

---

# End-to-End Workflow

```
User Question
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Generate Context
      │
      ▼
Insert Context into Prompt
      │
      ▼
Llama 3.1 Generates Response
      │
      ▼
Pydantic Validates Output
      │
      ▼
Return JSON Response
```

---

# Advantages of This Approach

- **Accurate Responses** – Answers are grounded in the provided documents.
- **Reduced Hallucination** – The LLM is instructed to rely only on retrieved context.
- **Fast Retrieval** – FAISS enables efficient semantic search.
- **Scalable** – New documents can be added without retraining the LLM.
- **Structured Output** – Responses follow a predefined JSON schema, making them easy to consume in applications.
- **Modular Design** – LangChain's runnable pipeline allows individual components to be modified or extended independently.
