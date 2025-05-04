# Building and Evaluating RAG Pipelines - Transcript

- this was a misfire when using Gemini Deep Research, but I loved the visibility into their Text-To Speech process

## **\<start\_of\_audio\> id: overview title: Audio Overview of Building and Evaluating RAG Pipelines**

\<start\_of\_audio\> audio\_overview

duration: 5:00

script: |

This audio overview summarizes the experiential journey designed for students to understand and implement key concepts in building and evaluating Retrieval-Augmented Generation (RAG) pipelines.

The journey begins with an introduction to the limitations of Large Language Models (LLMs) and how RAG addresses these by integrating external knowledge retrieval. It highlights the significance of effective document chunking strategies and the necessity of rigorous evaluation for RAG systems.

Phase 1 focuses on building a baseline RAG application using LangGraph. Students will learn the core steps of a minimal RAG pipeline: loading data, splitting it into chunks using a naive strategy like RecursiveCharacterTextSplitter, storing these chunks in a vector store, retrieving relevant chunks based on a query, and generating an answer using an LLM. They'll also be introduced to LangGraph for orchestrating this pipeline as a stateful graph.

In Phase 2, the focus shifts to evaluating this baseline using RAGAS, a framework specifically designed for RAG assessment. Students will understand the need for RAG-specific metrics beyond traditional NLP evaluation. The overview explains key RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness. It details the process of setting up an evaluation dataset and executing the RAGAS evaluation, emphasizing the interpretation of baseline results to identify areas for improvement.

Phase 3 delves into implementing a more advanced chunking strategy: semantic chunking using LangChain's SemanticChunker. The overview contrasts naive and semantic chunking, explains the embedding-based semantic chunking algorithm, and guides students through the practical implementation using LangChain, highlighting key parameters and considerations.

Phase 4 involves building and evaluating a second RAG application, this time incorporating the semantic chunking strategy within the LangGraph framework. Students will learn how to modify the pipeline to use the SemanticChunker, re-index the data, and then run the same RAGAS evaluation as in Phase 2 on this semantically enhanced pipeline. The overview emphasizes the importance of comparing these new results with the baseline.

Finally, Phase 5 centers on a comprehensive comparison of the results obtained with naive and semantic chunking. Students will analyze the impact of the chunking strategy on the RAGAS metrics, considering the trade-offs in complexity and performance. The overview provides in-depth elaboration on factors influencing the effectiveness of chunking and discusses the limitations of both semantic chunking and RAGAS evaluation. It concludes with detailed recommendations for choosing chunking strategies and suggests avenues for further exploration, encouraging students to experiment with different parameters, models, and advanced RAG techniques.

## **Throughout this journey, students will gain a practical understanding of the RAG pipeline, the critical role of document chunking, and the importance of systematic evaluation using specialized tools like RAGAS. They will experience firsthand how different design choices impact the performance of a RAG system.**
