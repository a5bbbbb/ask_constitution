# ReZanAi

ReZanAi is a Retrieval-Augmented Generation (RAG) based AI model designed to provide accurate, context-aware advice and interpretation of the Constitution of the Republic of Kazakhstan. By combining powerful language models with a custom retrieval system over constitutional text, ReZanAi delivers precise, citation-backed answers to legal and civic queries.

---

## Features

* **Contextual Retrieval**: Dynamically fetches relevant constitutional articles and commentary based on user queries.
* **Augmented Generation**: Leverages state-of-the-art LLMs to generate clear, well-structured explanations and advice.
* **Citation Support**: Integrates direct references to constitutional articles and amendments in responses for transparency.
* **Multi‑Language Interface**: Supports both Kazakh and English queries and responses.
* **Extensible Data Sources**: Easily plug in additional legal texts, amendments, or scholarly commentary.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Architecture Overview](#architecture-overview)
6. [RAG Implementation Details](#rag-implementation-details)
7. [Data Sources](#data-sources)
8. [Fine-Tuning the Model](#fine-tuning-the-model)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

---

## Getting Started

Follow these instructions to set up a local development environment and run ReZanAi.

### Prerequisites

* Python 3.9+
* Git
* [Poetry](https://python-poetry.org/) or pip
* API key for your chosen LLM provider (e.g., OpenAI, Anthropic)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your_org/ReZanAi.git
   cd ReZanAi
   ```

2. **Install dependencies**

   ```bash
   # Using Poetry
   poetry install

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env to add your LLM_API_KEY and any other settings
   ```

---

## Usage

To start the interactive query interface:

```bash
python run_app.py
```

Example:

```bash
> What does Article 33 say about language policy?
ReZanAi: Article 33 of the Constitution of Kazakhstan guarantees the right to choose the language of instructions and communication. It states that Kazakh is the state language, but everyone has the right to use any language they choose in private and public settings.
```

---

## Architecture Overview

ReZanAi comprises three main components:

1. **Document Store**: A vector database indexing all articles of the Constitution and supplemental commentary.
2. **Retrieval Module**: Queries the vector store to retrieve top‑k relevant passages for a given question.
3. **Generation Module**: An LLM that synthesizes retrieved passages into a coherent response, appending citations.

---

## RAG Implementation Details

* **Embedding Model**: Uses [SentenceTransformers](https://www.sbert.net/) to encode constitutional text into embeddings.
* **Vector Database**: Supports both FAISS (local) and Weaviate or Pinecone (managed).
* **Retriever Settings**:

  * Similarity Metric: Cosine distance
  * Top‑k Retrieval: Configurable (default: k = 5)
* **LLM Prompts**: Structured prompt template that includes:

  1. System instructions outlining role and citation format.
  2. Retrieved passages with inline article references.
  3. User query.

---

## Data Sources

* **Primary Text**: Official Constitution of the Republic of Kazakhstan (1995, amended 2022).
* **Language Versions**: Both Kazakh and English translations.
* **Supplemental Commentary**: Select scholarly articles and government explanations.

---

## Fine-Tuning the Model

If you wish to fine-tune the LLM on legal Q\&A data, follow these steps:

1. Prepare a JSONL file of `{"prompt": ..., "completion": ...}` pairs.
2. Run:

   ```bash
   openai api fine_tunes.create -t legal_qa.jsonl -m <base-model>
   ```
3. Update `MODEL_NAME` in `.env` to your fine-tuned model.

---

## API Reference

| Endpoint       | Method | Description                    |
| -------------- | ------ | ------------------------------ |
| `/query`       | POST   | Submit a constitutional query. |
| `/healthcheck` | GET    | Service status check.          |

**POST /query**

* **Request Body**:

  ```json
  { "question": "What rights does Article 34 guarantee?" }
  ```
* **Response**:

  ```json
  {
    "answer": "Article 34 guarantees the right to freedom of religion and worship, subject to laws of public order. [Art. 34]",
    "citations": ["Article 34"]
  }
  ```

---

## Contributing

We welcome contributions! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Refer to `CONTRIBUTING.md` for detailed guidelines.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

* SentenceTransformers team
* OpenAI for LLM access
* The Government of the Republic of Kazakhstan for constitutional texts

---

**ReZanAi** © 2025 Your Organization. All rights reserved.
