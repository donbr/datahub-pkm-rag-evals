# RAG Evaluations

Welcome to my exploration of Retrieval-Augmented Generation (RAG) evaluation techniques and chunking strategies. This site documents my journey understanding how different approaches to document chunking affect RAG system performance.

Start with the [Introduction](intro.md) to begin exploring!

![[rag-evals.png]]

## Quick Links

- [Chunking Strategies](rag-chunking-strategies.md) - Comparing naive and semantic approaches to document chunking
- [Evaluation with RAGAS](rag-evaluation-with-ragas.md) - Using specialized metrics for RAG assessment
- [Chunking Strategies Transcript](rag-chunking-strategies-transcript.md) - Audio transcript overview of the RAG evaluation journey

## Overview

The site provides a comprehensive walkthrough of:

- **RAG Chunking Strategies**: Comparing naive approaches (like RecursiveCharacterTextSplitter) with semantic chunking methods that preserve meaning
- **Evaluation with RAGAS**: Using specialized metrics like Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness
- **Chunking Strategies Transcript**: Audio overview summarizing the experiential journey through building and evaluating RAG pipelines

## Key Findings

- Semantic chunking significantly improves context precision and answer relevancy compared to naive approaches
- Different chunking strategies are optimal for different types of queries and content
- Trade-offs exist between processing speed and retrieval quality - semantic chunking improves quality but requires more computational resources
- Specialized evaluation frameworks like RAGAS provide insights traditional metrics miss, allowing precise measurement of RAG performance

## License

This content is available under the MIT License.
