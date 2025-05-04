# A Quick Intro

Welcome to my exploration of Retrieval-Augmented Generation (RAG) evaluation techniques and chunking strategies, capturing my journey to understand how different approaches to document chunking affect RAG system performance.

![[rag-evals.png]]

## Overview

- [Chunking Strategies]((rag-chunking-strategies.md)): Comparing naive approaches (like RecursiveCharacterTextSplitter) with semantic chunking methods that preserve meaning
- [Chunking Strategies Transcript](rag-chunking-strategies-transcript.md): Audio overview summarizing the experiential journey of building and evaluating RAG pipelines
- [Evaluation with RAGAS](rag-evaluation-with-ragas.md): Using specialized metrics like Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness

## Key Findings

- Semantic chunking significantly improves context precision and answer relevancy compared to naive approaches
- Different chunking strategies are optimal for different types of queries and content
- Trade-offs exist between processing speed and retrieval quality - semantic chunking improves quality but requires more computational resources
- Specialized evaluation frameworks like RAGAS provide insights traditional metrics miss, allowing precise measurement of RAG performance

## License

This content is available under the MIT License.
