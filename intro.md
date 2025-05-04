# My RAG Evaluations Journey

## Introduction

Welcome to my exploration of Retrieval-Augmented Generation (RAG) evaluation techniques and chunking strategies. This site documents my journey understanding how different approaches to document chunking affect RAG system performance, and how specialized evaluation metrics can help measure these effects systematically.

## Table of Contents

1. [Overview](#overview)
2. [Knowledge Areas](#knowledge-areas)
3. [Latest Insights](#latest-insights)
4. [Tools and Techniques](#tools-and-techniques)
5. [Resources](#resources)
6. [Next Steps](#next-steps)

---

## Overview

Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant information from external knowledge sources before generating responses. This approach addresses the limitations of LLMs related to knowledge cutoffs, hallucinations, and reasoning over specific domains.

Two critical aspects of RAG systems are:

1. **Chunking strategies** - How documents are split into manageable pieces for indexing and retrieval
2. **Evaluation methods** - How we measure the effectiveness of our RAG system in providing accurate, relevant, and faithful answers

This site documents my experiments with different chunking strategies and their impact on various evaluation metrics, helping to build more effective RAG systems.

---

## Knowledge Areas

This exploration covers several key knowledge areas:

- **[Chunking Strategies](rag-chunking-strategies.md)**: Comparing naive approaches like recursive character splitting with semantic chunking methods
- **[RAG Evaluation](rag-evaluation-with-ragas.md)**: Using the RAGAS framework to measure metrics like faithfulness, answer relevancy, and context precision
- **[LangGraph Integration](rag-with-langgraph.md)**: Building RAG pipelines as stateful graphs for better orchestration and control
- **[Results Comparison](rag-chunking-evaluation-results.md)**: Analyzing the impact of chunking strategies on evaluation metrics
- **Vector Databases**: Working with vector stores to efficiently index and retrieve document chunks
- **LLM Prompting**: Crafting effective prompts for context-aware generation using retrieved information

---

## Latest Insights

### Key Findings

- **Semantic Chunking Benefits**: Significant improvements in context precision (+27.4%) and answer relevancy (+16.4%) when using semantic chunking
- **Performance Trade-offs**: Semantic chunking increases processing time but delivers better quality results for complex queries
- **Query Type Impact**: Explanatory and multi-concept queries benefit most from semantic chunking approaches
- **Evaluation Framework**: RAGAS provides specialized metrics that capture unique aspects of RAG performance beyond traditional NLP metrics

### Recent Experiments

- **Naive vs. Semantic Chunking**: Conducted side-by-side comparison using identical test queries and evaluation metrics
- **LangGraph Implementation**: Built two parallel RAG pipelines using LangGraph for consistent comparison
- **Hybrid Approaches**: Early testing of combined strategies that select chunking method based on document complexity

---

## Tools and Techniques

### RAG Building Blocks

- **Text Splitting**: LangChain text splitters for both recursive character and semantic approaches
- **Vector Stores**: Chroma DB for embedding storage and retrieval
- **Embedding Models**: OpenAI's text-embedding-ada-002 for vector representations
- **LLMs**: GPT-4 for answer generation

### Evaluation Methods

- **RAGAS Framework**: Specialized metrics for RAG evaluation
- **A/B Testing**: Systematic comparison of different chunking strategies
- **Performance Monitoring**: Response time and computational resource tracking

---

## Resources

- **[Audio Overview](rag-audio-overview.md)**: Comprehensive audio guide to building and evaluating RAG pipelines
- **[Building with LangGraph](rag-with-langgraph.md)**: Building stateful RAG pipelines
- **[Chunking Strategy Comparison](rag-chunking-strategies.md)**: Comparing naive and semantic approaches
- **[Evaluation with RAGAS](rag-evaluation-with-ragas.md)**: Specialized metrics for RAG systems
- **[Results Analysis](rag-chunking-evaluation-results.md)**: Comparing performance across chunking strategies

---

## Next Steps

After exploring basic and semantic chunking, I plan to experiment with:

- Advanced chunking techniques like sliding window approaches
- Multi-vector retrieval methods
- Hybrid search combining semantic and keyword-based retrieval
- Fine-tuning of LLMs specifically for RAG applications
- Integration of structured knowledge alongside unstructured text

