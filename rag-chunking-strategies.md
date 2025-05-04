# Building and Evaluating RAG Pipelines

## **Part 1: Introduction - Setting the Stage for Advanced RAG**

Large Language Models (LLMs) possess remarkable capabilities in understanding and generating human language, trained on vast datasets.1 However, their knowledge is inherently static, limited to the data available up to their training cutoff date, and they lack access to private or domain-specific information.1 This can lead to outdated responses, factual inaccuracies (often termed "hallucinations"), or an inability to answer questions requiring specialized knowledge.4

Retrieval-Augmented Generation (RAG) emerges as a powerful architectural approach to mitigate these limitations.1 RAG enhances LLMs by integrating an information retrieval component.5 When presented with a query, a RAG system first retrieves relevant information from an external knowledge base (like a set of documents, databases, or websites) and then provides this retrieved context, along with the original query, to the LLM to generate a more informed, accurate, and contextually relevant response.1 This process effectively grounds the LLM's generation in verifiable facts, reducing hallucinations and allowing the model to leverage up-to-date or proprietary information without costly retraining.4

Building robust RAG systems involves orchestrating multiple components: data loading, processing (chunking), embedding, storage, retrieval, and generation. LangChain provides a comprehensive suite of tools for building such applications, offering modules for each step.1 LangGraph, an extension of LangChain, provides a more powerful and flexible framework for orchestrating complex, stateful, and potentially cyclic workflows, making it ideal for building sophisticated RAG agents.20 LangGraph allows for defining workflows as graphs, where nodes represent computational steps (like retrieving documents or generating an answer) and edges represent the flow of information and control logic.23

A critical aspect influencing RAG performance is the **chunking strategy** – the method used to break down large documents into smaller, manageable pieces for embedding and retrieval.26 The way documents are chunked directly impacts the relevance and completeness of the information retrieved, and consequently, the quality of the final generated answer.30 Naive methods, like fixed-size or simple recursive splitting, are easy to implement but may arbitrarily cut off context.30 More advanced techniques, like semantic chunking, aim to divide text based on meaning, potentially leading to more coherent and relevant chunks.26

However, building a RAG system is only half the battle. Rigorous **evaluation** is paramount to understand performance, identify weaknesses, and make informed improvements.35 Evaluating RAG is complex because it involves assessing both the retrieval component (Did we fetch the right information?) and the generation component (Did the LLM use the information correctly and answer the question well?).35 Frameworks like Ragas provide standardized metrics specifically designed for RAG evaluation, allowing for quantitative assessment of different pipeline configurations.36 Ragas utilizes LLMs as judges to evaluate aspects like the faithfulness of the answer to the retrieved context and the relevance of the answer to the original question, often without needing pre-annotated ground truth answers.38

This report provides a practical, step-by-step walkthrough designed as an experiential journey for students. We will:

1. Build a baseline RAG application using LangGraph with a naive chunking strategy (RecursiveCharacterTextSplitter).  
2. Evaluate this baseline using key RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness).  
3. Implement a more advanced semantic chunking strategy using LangChain SemanticChunker.  
4. Build a second RAG application using LangGraph with semantic chunking.  
5. Evaluate the semantic RAG pipeline using the same RAGAS metrics.  
6. Compare and contrast the results, analyzing the impact of the chunking strategy on overall RAG performance.

Through this process, students will gain hands-on experience in building, evaluating, and iteratively improving RAG systems, understanding the interplay between component choices (like chunking) and measurable performance outcomes.

## **Part 2: Phase 1 - Building the Baseline LangGraph RAG Application**

The foundation of our exploration is a baseline RAG application. This initial version utilizes standard LangChain components orchestrated by LangGraph, employing a straightforward "naive" chunking strategy. This serves as our starting point for comparison later. The core RAG process involves several distinct stages: Loading data, Splitting it into manageable chunks, Storing these chunks for efficient retrieval, Retrieving relevant chunks based on a query, and Generating a final answer using the query and retrieved context.1

### **2.1. The Minimal RAG Pipeline Steps**

1. **Load:** Data ingestion is the first step. We use LangChain's DocumentLoaders to read data from various sources (e.g., web pages, PDFs, text files) into a standardized Document format.1 For instance, WebBaseLoader can fetch and parse HTML content from a URL.18  
2. **Split:** Large documents often exceed LLM context windows and are inefficient to search. TextSplitters break these documents into smaller chunks.1 The goal is to create chunks small enough for processing but large enough to retain semantic meaning.26 For our baseline, we use the RecursiveCharacterTextSplitter, a common choice for generic text.18 It attempts to split text hierarchically based on a list of separators (like newlines, spaces) to keep meaningful units together as much as possible.46  
3. **Store:** The document chunks need to be indexed for efficient searching. This typically involves using an Embeddings model to convert the text chunks into numerical vector representations that capture their semantic meaning.1 These vectors are then stored in a VectorStore, a specialized database optimized for similarity searches.1 LangChain supports numerous embedding models (like OpenAI's) and vector stores (like FAISS, Milvus, Qdrant, Chroma).18  
4. **Retrieve:** When a user query arrives, a Retriever interacts with the VectorStore. It embeds the user query using the same embedding model and searches the vector store for chunks with the most similar embeddings (semantic similarity).1 The retriever returns these relevant chunks.  
5. **Generate:** Finally, a ChatModel or LLM generates an answer. It receives a prompt containing the original user query and the content of the retrieved document chunks.1 The LLM uses this augmented context to synthesize a relevant and factually grounded response.5

### **2.2. Orchestration with LangGraph**

While LangChain provides the building blocks, LangGraph allows us to define this RAG pipeline as a stateful graph, enabling more complex control flow, cycles (like re-querying if results are poor), and easier integration of components like agents or human-in-the-loop checks.21

Defining the State:  
The state is the memory or the information that flows through the graph. For a basic RAG pipeline, the state typically needs to hold, at minimum, the user's question and the documents retrieved. A more robust state might track the conversation history, intermediate steps, or generated answers. LangGraph uses Python's TypedDict to define the state structure.25

```python

from typing import List, TypedDict  
from typing\_extensions import Annotated  
from langgraph.graph.message import add\_messages

\# Define the state structure for our RAG graph  
class RAGState(TypedDict):  
    \# Stores the history of messages (user query, AI responses)  
    messages: Annotated\[List, add\_messages\]  
    \# Stores the documents retrieved by the retriever node  
    documents: List\[Document\] \# Assuming Document is imported from langchain\_core.documents
```

Defining the Nodes:  
Nodes represent the computational steps or functions in our graph. For a minimal RAG:

* **retrieve Node:** This node takes the latest user query from the state, uses the Retriever (configured with our vector store) to fetch relevant documents, and updates the state with these documents.  
* **generate Node:** This node takes the latest query and the retrieved documents from the state, formats them into a prompt, calls the LLM to generate an answer, and updates the state with the AI's response message.  
* **(Optional) grade\_documents Node:** In more advanced RAG 22, a node could evaluate the relevance of retrieved documents before generation. For our baseline, we'll omit this.  
* **(Optional) rewrite\_query Node:** An agentic or adaptive RAG might rewrite the query if initial retrieval is poor.22 Omitted in baseline.

```python

\# Example Node Function (Conceptual)  
def retrieve\_documents(state: RAGState):  
    """Retrieves documents based on the latest query."""  
    latest\_query \= state\['messages'\]\[-1\].content \# Assuming message format  
    \# retriever is pre-configured (e.g., retriever \= vectorstore.as\_retriever())  
    retrieved\_docs \= retriever.invoke(latest\_query)  
    return {"documents": retrieved\_docs}

def generate\_answer(state: RAGState):  
    """Generates an answer using the query and retrieved documents."""  
    query \= state\['messages'\]\[-1\].content  
    docs \= state\['documents'\]  
    \# prompt and llm are pre-configured  
    rag\_chain \= prompt | llm | StrOutputParser() \# Example chain  
    generation \= rag\_chain.invoke({"context": docs, "question": query})  
    return {"messages": \[AIMessage(content=generation)\]} \# Append AI message
```

Defining the Edges:  
Edges define the sequence and conditional logic connecting the nodes.

* **Entry Point:** The graph needs a starting point, typically linked to the first node that processes the initial user input (often the retrieve node after the user query is added to the state). LangGraph uses START for this.25  
* **Sequential Flow:** An edge connects the retrieve node to the generate node, indicating that generation happens after retrieval.  
* **Conditional Edges (Optional):** More complex graphs use conditional edges to route the flow based on the state. For example, after a grade\_documents node, an edge might decide whether to proceed to generate or loop back to rewrite\_query.25 The baseline is linear.  
* **End Point:** The graph needs an end state, END.25

```python

from langgraph.graph import StateGraph, END, START

\# Initialize the graph builder  
workflow \= StateGraph(RAGState)

\# Add nodes  
workflow.add\_node("retrieve", retrieve\_documents)  
workflow.add\_node("generate", generate\_answer)

\# Define edges  
workflow.set\_entry\_point("retrieve") \# Start with retrieval  
workflow.add\_edge("retrieve", "generate") \# After retrieval, generate  
workflow.add\_edge("generate", END) \# After generation, end

\# Compile the graph  
app \= workflow.compile()
```

This compiled app represents our executable LangGraph RAG pipeline. It accepts an initial state (containing the user query) and streams the state updates as it progresses through the defined nodes and edges. This baseline structure provides a functional RAG system ready for evaluation. The explicit state management and graph structure offered by LangGraph make it easier to visualize, debug (especially with LangSmith integration 20), and extend compared to simple sequential chains.23

## **Part 3: Phase 2 - Baseline Evaluation with RAGAS**

Building a RAG system is the first step; understanding its effectiveness is the crucial next one.35 Evaluation helps identify weaknesses, compare different configurations (like chunking strategies), and ultimately build more reliable and trustworthy AI applications.35 Traditional NLP metrics often fall short for RAG, as they don't capture the nuances of retrieval quality and factual grounding.35

### **3.1. The Need for RAG-Specific Evaluation**

Evaluating RAG systems presents unique challenges:

* **Dual Component Assessment:** Poor performance can stem from faulty retrieval (irrelevant or incomplete context) or flawed generation (hallucinations, poor use of context).35 Evaluation needs to pinpoint the source of failure.  
* **Retrieval Quality:** Simple relevance scores might not capture if the retrieved content, although topically related, is actually *useful* for answering the specific query.35  
* **Factual Grounding (Faithfulness):** The LLM must generate responses based *only* on the provided context, avoiding reliance on its internal, potentially outdated or incorrect, knowledge.4 Metrics must assess this grounding.

### **3.2. Introduction to RAGAS Metrics**

Ragas is an open-source framework designed specifically for evaluating RAG pipelines, often without requiring ground truth human annotations for all metrics.38 It uses LLMs as judges to assess different quality dimensions.41 We will focus on five key metrics:

1. **Faithfulness:** Measures the factual consistency of the generated answer with the retrieved contexts. It calculates the ratio of claims in the answer that are directly supported by the context to the total number of claims.39 A high score indicates the answer is well-grounded and avoids hallucination based on the provided context. It's crucial to note that faithfulness *does not* measure factual accuracy against the real world, only against the provided context.57  
   * *Calculation:* Identify claims in the answer -\> Verify each claim against context -\> Score \= (Supported Claims) / (Total Claims).56  
2. **Answer Relevancy:** Assesses how pertinent the generated answer is to the input question. It penalizes answers that are incomplete or contain redundant information, focusing on conciseness and directness.41 It does *not* consider factuality.58  
   * *Calculation:* Generate 'n' plausible questions from the answer using an LLM -\> Calculate the average cosine similarity between the embeddings of these generated questions and the original question embedding.57 A high score suggests the answer closely matches the intent of the original question.  
3. **Context Precision:** Evaluates the signal-to-noise ratio in the retrieved contexts. It measures whether the most relevant chunks are ranked higher by the retriever.41 A high score indicates the retriever is effective at prioritizing useful information.  
   * *Calculation:* Uses an LLM to identify relevant sentences/chunks within the contexts relative to the question. It then calculates a precision score, often based on the rank of these relevant chunks (e.g., using average precision@k logic).59  
4. **Context Recall:** Measures the extent to which the retrieved contexts contain all the information necessary to answer the question, based on a ground truth answer.41 This is often the only metric requiring a reference (ground\_truth) answer.  
   * *Calculation:* Uses an LLM to compare the ground\_truth answer with the contexts. It identifies sentences/claims in the ground truth and checks if they can be attributed to the retrieved context. Score \= (Attributed Statements) / (Total Statements in Ground Truth).61  
5. **Answer Correctness:** Provides a more holistic measure of the answer's quality compared to a ground\_truth answer. It combines aspects of factual correctness (is the information accurate based on the ground truth?) and semantic similarity (does the answer convey the same meaning as the ground truth?).61  
   * *Calculation:* A weighted combination of factuality (comparing claims in the generated answer vs. ground truth) and semantic similarity (using embedding models or cross-encoders).61 Default weights are often 75% factuality, 25% similarity.61

These metrics provide a multi-faceted view of RAG performance, assessing both retrieval effectiveness (Context Precision, Context Recall) and generation quality (Faithfulness, Answer Relevancy, Answer Correctness).41

### **3.3. Evaluation Setup: Preparing the Dataset**

To run RAGAS, we need an evaluation dataset. This dataset typically consists of question-answer pairs, along with the context retrieved by the RAG system for each question.51

1. **Define Questions:** Create a list of representative questions (sample\_queries) that cover the expected use cases and knowledge domain of your RAG application.60  
2. **Define Ground Truth Answers:** For metrics like Context Recall and Answer Correctness, provide the ideal or correct answer (expected\_responses or ground\_truth) for each question.51 Generating this "golden dataset" can be time-consuming but is crucial for certain evaluations.35 Synthetic data generation using LLMs is an alternative approach.65  
3. **Run RAG Pipeline:** Execute the baseline LangGraph RAG application (built in Phase 1\) for each question in your test set.  
4. **Collect Results:** For each question, store the input question (user\_input), the retrieved\_contexts, the generated answer (response), and the ground\_truth (reference) answer.63  
5. **Format for RAGAS:** Structure this collected data as a list of dictionaries or a Hugging Face Dataset, where each entry contains the keys: "user\_input", "retrieved\_contexts", "response", and "reference" (or "ground\_truth").51  
6. **Load into EvaluationDataset:** Use EvaluationDataset.from\_list(dataset) or EvaluationDataset.from\_hf\_dataset() to load the data into the RAGAS format.63

### **3.4. Executing the RAGAS Evaluation**

With the prepared EvaluationDataset, running the RAGAS evaluation is straightforward:

1. **Import evaluate and Metrics:**  

   ```python  
   from ragas import evaluate  
   from ragas.metrics import (  
       faithfulness,  
       answer\_relevancy,  
       context\_precision,  
       context\_recall,  
       answer\_correctness \# Requires ground\_truth in dataset  
   )
   ```  
   *41*  

2. **Configure Evaluator LLM:** RAGAS uses an LLM for judging metrics like Faithfulness and Answer Relevancy. Wrap your chosen LLM (e.g., from OpenAI, Bedrock, Azure) using LangchainLLMWrapper.63  

   ```python  
   from ragas.llms import LangchainLLMWrapper  
   from langchain\_openai import ChatOpenAI \# Or your chosen LLM provider

   \# Ensure API keys are set as environment variables  
   evaluator\_llm \= LangchainLLMWrapper(ChatOpenAI(model="gpt-4o")) \# Example
   ```

3. **Run Evaluation:** Call evaluate(), passing the dataset, metrics list, and evaluator LLM.57  

   ```python  
   results \= evaluate(  
       dataset=evaluation\_dataset, \# Your loaded dataset object  
       metrics=\[  
           faithfulness,  
           answer\_relevancy,  
           context\_precision,  
           context\_recall,  
           answer\_correctness,  
       \],  
       llm=evaluator\_llm,  
       \# Add embeddings if needed for specific metrics like answer\_relevancy  
       \# embeddings=OpenAIEmbeddings() \# Example  
   )
   ```

4. **Analyze Results:** The results object contains the scores. You can convert it to a Pandas DataFrame for easier analysis: df \= results.to\_pandas().51

### **3.5. Interpreting Baseline Results**

Examine the scores in the resulting DataFrame (or dictionary). For the baseline (naive chunking):

* **Context Precision/Recall:** How well did the naive chunking strategy support retrieval? Low scores might indicate that the simple splitting method failed to isolate relevant information effectively or broke context across chunks.  
* **Faithfulness:** Are the answers grounded in the (potentially poorly) retrieved context? Even with bad retrieval, faithfulness might be high if the LLM simply stated it couldn't answer or only used the limited context provided. Low faithfulness despite retrieval suggests the LLM is hallucinating or ignoring the context.  
* **Answer Relevancy/Correctness:** How good are the final answers? These scores reflect the end-to-end performance. Poor scores here, potentially linked to low context scores, highlight the impact of the retrieval stage (and thus chunking) on the final output.

This baseline evaluation provides a quantitative benchmark. It reveals potential weaknesses in the naive approach, setting the stage for exploring whether a more sophisticated chunking strategy can yield better results. The dependency of generation quality (Faithfulness, Answer Correctness) on retrieval quality (Context Precision/Recall) is a key relationship to observe. Poor retrieval often limits the potential for good generation, regardless of the LLM's capabilities.

## **Part 4: Phase 3 - Implementing Semantic Chunking**

The baseline evaluation likely highlighted areas for improvement, particularly in the retrieval metrics (Context Precision/Recall), which are heavily influenced by how documents are split into chunks. Naive chunking strategies, like the RecursiveCharacterTextSplitter used in Phase 1, prioritize structure (separators, chunk size) over content meaning.26 While simple and fast, they can arbitrarily split semantically related sentences or paragraphs, leading to fragmented context during retrieval.30

### **4.1. Naive vs. Semantic Chunking**

* **Naive Chunking (e.g., RecursiveCharacterTextSplitter):**  
  * **Pros:** Simple, fast, easy to implement.30 Uses structural rules (separators like \\n\\n, \\n, ) and size limits (chunk\_size).47  
  * **Cons:** Ignores semantic content, can break sentences or related ideas across chunks, potentially diluting meaning in embeddings and leading to poor retrieval relevance.30 The primary control parameter is chunk\_size, a structural limit.47  
* **Semantic Chunking:**  
  * **Pros:** Aims to create chunks based on semantic meaning, grouping related sentences together.26 This can lead to more coherent chunks, potentially improving embedding quality and retrieval relevance, resulting in better RAG performance.30 It addresses the "signal-to-noise" problem by creating focused chunks.30  
  * **Cons:** More computationally intensive and slower than naive methods due to embedding calculations.30 Requires careful selection of an embedding model and tuning of semantic threshold parameters.34 May still struggle with certain content types (e.g., code, formulas) or produce overly large chunks if not configured carefully.31 Some studies suggest simpler sentence splitting can sometimes perform better, indicating semantic chunking isn't universally superior.43

The core idea behind semantic chunking is to divide text where the *meaning* shifts, rather than just where a character limit is reached or a specific separator appears.30

### **4.2. The Semantic Chunking Algorithm (Embedding-Based Approach)**

The most common approach to semantic chunking, implemented in LangChain's SemanticChunker, relies on embedding similarity 30:

1. **Split into Sentences:** The input text is first broken down into individual sentences, typically using punctuation as delimiters.29 This forms the initial granular units.  
2. **Embed Sentences:** Each sentence is then converted into a numerical vector (embedding) using a chosen embedding model (e.g., OpenAI, Hugging Face models).30 These embeddings capture the semantic meaning of each sentence.  
3. **Calculate Distances:** The algorithm calculates the semantic distance (often cosine distance, where distance \= 1 - cosine similarity) between the embeddings of consecutive sentences.30 A small distance implies high semantic similarity, meaning the sentences likely discuss related topics.  
4. **Identify Breakpoints:** A threshold is applied to these distances to determine where chunk boundaries should occur.30 If the distance between two adjacent sentences exceeds the threshold, it signifies a potential shift in topic, and a breakpoint is marked.  
5. **Group Sentences:** Sentences between identified breakpoints are grouped together to form the final chunks.30 This ensures that sentences within a chunk are semantically cohesive.

### **4.3. Implementing Semantic Chunking with LangChain**

LangChain provides the SemanticChunker class (currently in langchain\_experimental) for this purpose.46

1. **Installation:** Ensure langchain\_experimental and an embedding provider library (e.g., langchain\_openai) are installed.  
   Bash  
   pip install langchain\_experimental langchain\_openai

2. **Instantiation:** Create an instance of SemanticChunker, providing an embedding model instance.74  

   ```python  
   from langchain\_experimental.text\_splitter import SemanticChunker  
   from langchain\_openai.embeddings import OpenAIEmbeddings \# Or your chosen embeddings

   embeddings \= OpenAIEmbeddings()  
   semantic\_text\_splitter \= SemanticChunker(  
       embeddings,  
       breakpoint\_threshold\_type="percentile", \# Or "standard\_deviation", "interquartile", "gradient"  
       breakpoint\_threshold\_amount=95 \# Adjust threshold value as needed  
   )
   ```

3. **Key Parameters** 74:  
   * embeddings: (Required) An instance of a LangChain Embeddings class (e.g., OpenAIEmbeddings, HuggingFaceInferenceAPIEmbeddings 79). This model is used internally to calculate sentence similarities.  
   * breakpoint\_threshold\_type: (Optional, defaults to "percentile") Specifies the method used to determine the split threshold based on calculated distances between sentences. Options include:  
     * "percentile": Splits when the distance exceeds the Nth percentile of all distances.  
     * "standard\_deviation": Splits when the distance exceeds N standard deviations from the mean distance.  
     * "interquartile": Splits based on the interquartile range (IQR) of distances.  
     * "gradient": Uses gradient changes in distances, potentially better for highly correlated text.  
   * breakpoint\_threshold\_amount: (Optional, defaults depend on type, e.g., 95.0 for percentile) The numerical value used with the chosen breakpoint\_threshold\_type. For "percentile", it's a value between 0 and 100\. Adjusting this tunes the sensitivity – a lower percentile (e.g., 80\) will likely create more, smaller chunks, while a higher percentile (e.g., 98\) will create fewer, larger chunks.34

Unlike the chunk\_size parameter in naive chunking which sets a hard structural limit, the breakpoint\_threshold\_type and breakpoint\_threshold\_amount parameters in semantic chunking directly control the *semantic* boundaries based on meaning shifts detected via embeddings. This offers finer control over chunk coherence but necessitates a deeper understanding of the embedding space and likely requires experimentation and tuning to find the optimal settings for a specific dataset and task.30 There isn't a one-size-fits-all formula.30

Potential nuances to consider include the need for experimentation 30, the possibility of generating chunks larger than an embedding model's context window if the text is very homogenous 75, specific handling for code or tables 31, and the observation that simpler sentence splitting might sometimes yield better evaluation results.43

With the SemanticChunker instantiated, you can use its .split\_documents() or .create\_documents() method just like any other LangChain text splitter to process your loaded documents.

## **Part 5: Phase 4 - Building and Evaluating the Semantic RAG**

Having defined the semantic chunking strategy and its implementation using SemanticChunker, we now modify our baseline RAG pipeline to incorporate it. The goal is to assess whether this change in chunking strategy leads to a measurable improvement in performance according to our RAGAS metrics.

### **5.1. Modifying the Pipeline for Semantic Chunking**

The core LangGraph orchestration structure (state, nodes, edges defined in Phase 1\) can largely remain the same. The key change lies in *how* the documents are processed *before* being stored and retrieved.

1. **Replace Step 2 (Split):** Instead of using RecursiveCharacterTextSplitter, use the SemanticChunker instance created in Phase 3 (Section 4.3). Apply its .split\_documents() method to the loaded documents.  

   ```python  
   \# Assuming 'loaded\_docs' is the list of Document objects from the Load step  
   \# And 'semantic\_text\_splitter' is the configured SemanticChunker instance

   semantic\_chunks \= semantic\_text\_splitter.split\_documents(loaded\_docs)
   ```

2. **Re-run Step 3 (Store):** Create a *new* vector store index using these semantic\_chunks. It is crucial to use a different index name or collection name (depending on the vector store provider, e.g., Qdrant collection name 25) to keep the semantically chunked data separate from the baseline data stored in Phase 1\. This ensures we are comparing the effect of the chunking strategy alone.  

   ```python  
   \# Assuming 'vectorstore\_provider' is your chosen VectorStore class (e.g., Qdrant, FAISS)  
   \# And 'embeddings' is the same embedding model used before

   semantic\_vectorstore \= vectorstore\_provider.from\_documents(  
       documents=semantic\_chunks,  
       embedding=embeddings,  
       \# Add specific connection args for your vector store, e.g., URL, API key, collection\_name  
       collection\_name="rag\_semantic\_chunks\_v1" \# Use a distinct name  
   )
   ```

3. **Update Retriever Configuration:** Ensure the retrieve node function in your LangGraph definition now uses a retriever configured with this *new* semantic\_vectorstore.  

   ```python  
   \# Update the retriever used in the 'retrieve\_documents' node function  
   semantic\_retriever \= semantic\_vectorstore.as\_retriever()

   \# Modify the retrieve\_documents node function (or pass retriever via config)  
   \# to use 'semantic\_retriever' instead of the baseline retriever.
   ```

The generate node and the overall graph flow (app \= workflow.compile()) remain unchanged, but they will now operate on documents retrieved from the semantically chunked index.

### **5.2. Running the Semantic RAG Pipeline**

With the modified setup:

1. Instantiate the LangGraph application (app) compiled with the updated retrieve node configuration (using the semantic\_retriever).  
2. Use the *same evaluation dataset questions* (the sample\_queries or user\_input list from Phase 2, Section 3.3) as input.  
3. Run the pipeline for each question. This will involve the retrieve node fetching chunks from the semantic\_vectorstore and the generate node producing answers based on these semantically retrieved contexts.  
4. Collect the results: For each question, store the user\_input, the new retrieved\_contexts (from the semantic store), and the new response (generated based on semantic context). Keep the original reference (ground truth) answers.

### **5.3. Executing RAGAS Evaluation on Semantic RAG**

Now, evaluate the performance of this modified pipeline using the exact same RAGAS setup as in Phase 2 (Section 3.4).

1. **Prepare Dataset:** Format the results collected in Section 5.2 (user input, semantic contexts, semantic answers, ground truth) into the RAGAS EvaluationDataset structure.63  
2. **Run evaluate():** Use the same list of RAGAS metrics (faithfulness, answer\_relevancy, context\_precision, context\_recall, answer\_correctness) and the same evaluator\_llm configuration.  

   ```python  
   \# Assuming 'semantic\_evaluation\_dataset' is the dataset object with semantic RAG results  
   semantic\_results \= evaluate(  
       dataset=semantic\_evaluation\_dataset,  
       metrics=\[  
           faithfulness,  
           answer\_relevancy,  
           context\_precision,  
           context\_recall,  
           answer\_correctness,  
       \],  
       llm=evaluator\_llm  
       \# Add embeddings if needed  
   )
   ```

### **5.4. Interpreting Semantic RAG Results**

Analyze the semantic\_results (e.g., by converting to a DataFrame: semantic\_df \= semantic\_results.to\_pandas()).

* Observe the scores for each metric.  
* Qualitatively compare these scores to the baseline results obtained in Phase 2\. Did Context Precision/Recall improve? Did Faithfulness or Answer Correctness increase? Are there any metrics where performance decreased?

This evaluation provides the quantitative data needed for a direct comparison between the naive and semantic chunking strategies within the same RAG pipeline architecture and evaluation framework.

## **Part 6: Phase 5 - Comparison, Insights, and Conclusion**

The final phase involves directly comparing the performance of the baseline RAG (using naive RecursiveCharacterTextSplitter) and the modified RAG (using SemanticChunker), drawing conclusions about the impact of chunking, and suggesting avenues for further exploration.

### **6.1. Direct Comparison: Naive vs. Semantic Chunking Performance**

The core of the comparison lies in the RAGAS evaluation scores obtained in Phase 2 and Phase 4\.

**Table 1: RAGAS Evaluation Results Comparison**

| Metric | Naive Chunking RAG (Baseline) | Semantic Chunking RAG | Change |
| :---- | :---- | :---- | :---- |
| Faithfulness |  |  | \[Difference\] |
| Answer Relevancy |  |  | \[Difference\] |
| Context Precision |  |  | \[Difference\] |
| Context Recall |  |  | \[Difference\] |
| Answer Correctness |  |  | \[Difference\] |

*(Note: Replace bracketed placeholders with actual scores obtained during the practical execution)*

**Analysis of Results:**

* **Context Precision/Recall:** Examine the change in these retrieval-focused metrics. Semantic chunking aims to improve retrieval by creating more coherent chunks.30 An increase in Context Precision would suggest the retrieved chunks are more relevant (less noise), while an increase in Context Recall would indicate that more of the necessary information from the ground truth was successfully retrieved.41 Did semantic chunking deliver on this promise according to the scores? Sometimes, simpler sentence splitting (closer to naive methods if separators are primarily sentence-based) can outperform semantic chunking, especially if the document structure is already clear or if the semantic similarity measures introduce noise.43  
* **Faithfulness/Answer Relevancy/Answer Correctness:** Analyze how changes in retrieval quality impacted generation quality. Ideally, improved Context Precision/Recall should lead to higher scores in these metrics.73 If context scores improved but generation scores didn't (or worsened), it might point to issues with the LLM's ability to utilize the (potentially differently structured) semantic context, or limitations in the RAGAS metrics themselves. Conversely, if context scores worsened but generation scores improved, it might suggest the semantic chunks, while less precise/recalled according to RAGAS, were somehow more beneficial for the LLM's generation process in this specific case. This highlights the dependency within the RAG pipeline: generation quality is often capped by retrieval quality.35

### **6.2. In-depth Elaboration & Multi-layered Considerations**

Beyond the raw scores, several factors warrant discussion:

* **Complexity and Performance Trade-offs:** Semantic chunking involves embedding calculations during the splitting process, making it computationally more expensive and slower than naive methods.30 Furthermore, it introduces hyperparameters (breakpoint\_threshold\_type, breakpoint\_threshold\_amount) that require tuning based on the specific dataset and embedding model, adding complexity to the development process.34 The observed performance gains (or lack thereof) in Table 1 must be weighed against this increased setup cost and processing time. Is the improvement significant enough to justify the extra effort?  
* **Influence of Data Characteristics:** The effectiveness of any chunking strategy can depend heavily on the nature of the source documents.29 For well-structured documents with clear paragraphs that align with distinct topics, a naive RecursiveCharacterTextSplitter might perform surprisingly well, potentially approaching the effectiveness of semantic chunking.43 Conversely, for dense, unstructured text or documents covering multiple interleaved topics, semantic chunking might offer a more significant advantage by actively grouping related sentences.34  
* **Limitations of Semantic Chunking and RAGAS:** Semantic chunking is not a silver bullet. It can struggle with non-textual elements, code snippets, or mathematical formulas where standard text embeddings may not capture relevance accurately.31 It might also produce chunks exceeding LLM context limits if not carefully configured.75 Similarly, RAGAS metrics, while valuable, are approximations. They rely on LLM judgments which can have their own biases or inaccuracies, and metrics like Context Recall depend on the quality of the ground truth data.35 Evaluation should be seen as an informative guide, not an absolute measure of perfection.

### **6.3. Detailed Recommendations & Further Exploration**

Based on this experiential journey, students can draw practical conclusions:

* **When to Choose Naive Chunking:** Start with RecursiveCharacterTextSplitter for simplicity and speed, especially for well-structured documents or initial prototyping. Tune chunk\_size and chunk\_overlap based on the embedding model's context window and initial evaluation results.  
* **When to Choose Semantic Chunking:** Consider SemanticChunker if baseline evaluation reveals poor retrieval performance (low Context Precision/Recall) likely due to fragmented context, particularly with less structured or multi-topic documents. Be prepared to invest time in tuning embedding models and breakpoint thresholds.

**Suggestions for Further Student Exploration:**

1. **Tune Semantic Parameters:** Experiment with different breakpoint\_threshold\_type values ('standard\_deviation', 'interquartile', 'gradient') and adjust the breakpoint\_threshold\_amount in SemanticChunker. Evaluate each configuration with RAGAS to observe the impact.74  
2. **Vary Embedding Models:** Test how different embedding models (e.g., open-source sentence transformers via HuggingFaceInferenceAPIEmbeddings 79, other OpenAI models) affect both semantic chunking results and overall RAGAS scores.  
3. **Implement Chunk Overlap:** Modify the semantic chunking process or explore text splitters that allow adding overlap to semantic chunks to potentially improve context continuity.26  
4. **Explore Other Chunking Strategies:** Investigate and implement other methods mentioned in the research, such as document-specific splitters (Markdown, HTML) 26, agentic chunking 26, or hierarchical approaches.  
5. **Advanced LangGraph Patterns:** Leverage LangGraph's capabilities to build more sophisticated RAG workflows like Adaptive RAG, Corrective RAG, or Self-RAG, which incorporate feedback loops and dynamic adjustments based on retrieval quality.22  
6. **Deeper Evaluation:** Utilize additional RAGAS metrics (e.g., Context Utilization, Answer Similarity) or explore alternative evaluation frameworks like TruLens 81 or LLM-based evaluation benchmarks.36 Consider incorporating human evaluation for qualitative feedback.36

### **6.4. Concluding Thoughts**

This walkthrough demonstrated the process of building, evaluating, and comparing RAG pipelines using LangGraph and RAGAS, focusing on the critical role of the chunking strategy. We constructed a baseline RAG with naive chunking, established its performance using RAGAS metrics, implemented semantic chunking, and evaluated the modified pipeline.

The key takeaway is that the method chosen for splitting documents into chunks significantly influences the effectiveness of the retrieval process, which in turn impacts the quality and faithfulness of the generated answers. While semantic chunking offers a more sophisticated, meaning-aware approach that can potentially improve performance, especially for complex texts, it introduces additional complexity and requires careful tuning. Naive methods remain a viable starting point, particularly for well-structured data.

Ultimately, there is no single "best" chunking strategy. The optimal choice depends on the specific data, the application's requirements, and the acceptable trade-offs between performance, complexity, and computational cost. Rigorous, metric-driven evaluation, facilitated by frameworks like RAGAS, is essential for understanding these trade-offs, diagnosing issues within the RAG pipeline, and making informed decisions to build more robust, reliable, and effective AI systems.35 Combining structured development tools like LangGraph with systematic evaluation practices empowers developers and researchers to navigate the complexities of RAG and unlock its full potential.

#### **Works cited**

1. Q\&A with RAG - ️ LangChain, accessed May 4, 2025, [https://python.langchain.com/v0.1/docs/use\_cases/question\_answering/](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)  
2. What is Retrieval Augmented Generation (RAG)? - Databricks, accessed May 4, 2025, [https://www.databricks.com/glossary/retrieval-augmented-generation-rag](https://www.databricks.com/glossary/retrieval-augmented-generation-rag)  
3. Retrieval Augmented Generation (RAG) - Pinecone, accessed May 4, 2025, [https://www.pinecone.io/learn/retrieval-augmented-generation/](https://www.pinecone.io/learn/retrieval-augmented-generation/)  
4. Ingest-And-Ground: Dispelling Hallucinations from Continually-Pretrained LLMs with RAG, accessed May 4, 2025, [https://arxiv.org/html/2410.02825v2](https://arxiv.org/html/2410.02825v2)  
5. What is Retrieval-Augmented Generation (RAG)? | Google Cloud, accessed May 4, 2025, [https://cloud.google.com/use-cases/retrieval-augmented-generation](https://cloud.google.com/use-cases/retrieval-augmented-generation)  
6. What is retrieval-augmented generation (RAG)? - IBM Research, accessed May 4, 2025, [https://research.ibm.com/blog/retrieval-augmented-generation-RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)  
7. What is Retrieval-Augmented Generation (RAG)? A Practical Guide - K2view, accessed May 4, 2025, [https://www.k2view.com/what-is-retrieval-augmented-generation](https://www.k2view.com/what-is-retrieval-augmented-generation)  
8. Retrieval augmented generation (rag) - LangChain.js, accessed May 4, 2025, [https://js.langchain.com/docs/concepts/rag/](https://js.langchain.com/docs/concepts/rag/)  
9. What is RAG? - Retrieval-Augmented Generation AI Explained - AWS, accessed May 4, 2025, [https://aws.amazon.com/what-is/retrieval-augmented-generation/](https://aws.amazon.com/what-is/retrieval-augmented-generation/)  
10. Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation - arXiv, accessed May 4, 2025, [https://arxiv.org/html/2502.15040v1](https://arxiv.org/html/2502.15040v1)  
11. Reducing hallucination in structured outputs via Retrieval-Augmented Generation - arXiv, accessed May 4, 2025, [https://arxiv.org/html/2404.08189v1](https://arxiv.org/html/2404.08189v1)  
12. \[2410.02825\] Ingest-And-Ground: Dispelling Hallucinations from Continually-Pretrained LLMs with RAG - arXiv, accessed May 4, 2025, [https://arxiv.org/abs/2410.02825](https://arxiv.org/abs/2410.02825)  
13. arXiv:2502.17125v1 \[cs.CL\] 24 Feb 2025, accessed May 4, 2025, [https://arxiv.org/pdf/2502.17125?](https://arxiv.org/pdf/2502.17125)  
14. LettuceDetect: A Hallucination Detection Framework for RAG Applications - arXiv, accessed May 4, 2025, [https://arxiv.org/html/2502.17125v1](https://arxiv.org/html/2502.17125v1)  
15. A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning through RAG and Incremental Knowledge Graph Learning Integration - arXiv, accessed May 4, 2025, [https://arxiv.org/html/2503.13514v1](https://arxiv.org/html/2503.13514v1)  
16. A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning through RAG and Incremental Kn - arXiv, accessed May 4, 2025, [https://arxiv.org/pdf/2503.13514](https://arxiv.org/pdf/2503.13514)  
17. Retrieval-augmented generation - Wikipedia, accessed May 4, 2025, [https://en.wikipedia.org/wiki/Retrieval-augmented\_generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)  
18. Retrieval-Augmented Generation (RAG) with Milvus and LangChain, accessed May 4, 2025, [https://milvus.io/docs/integrate\_with\_langchain.md](https://milvus.io/docs/integrate_with_langchain.md)  
19. Retrieval - ️ LangChain, accessed May 4, 2025, [https://python.langchain.com/v0.1/docs/modules/data\_connection/](https://python.langchain.com/v0.1/docs/modules/data_connection/)  
20. Build a Retrieval Augmented Generation (RAG) App: Part 1 - LangChain.js, accessed May 4, 2025, [https://js.langchain.com/docs/tutorials/rag/](https://js.langchain.com/docs/tutorials/rag/)  
21. LangGraph - LangChain, accessed May 4, 2025, [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)  
22. Tutorials - GitHub Pages, accessed May 4, 2025, [https://langchain-ai.github.io/langgraph/tutorials/](https://langchain-ai.github.io/langgraph/tutorials/)  
23. LangGraph - GitHub Pages, accessed May 4, 2025, [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)  
24. Complete Guide to Building LangChain Agents with the LangGraph Framework - Zep, accessed May 4, 2025, [https://www.getzep.com/ai-agents/langchain-agents-langgraph](https://www.getzep.com/ai-agents/langchain-agents-langgraph)  
25. Agentic RAG With LangGraph - Qdrant, accessed May 4, 2025, [https://qdrant.tech/documentation/agentic-rag-langgraph/](https://qdrant.tech/documentation/agentic-rag-langgraph/)  
26. Chunking strategies for RAG tutorial using Granite - IBM, accessed May 4, 2025, [https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai](https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai)  
27. 15 Chunking Techniques to Build Exceptional RAGs Systems - Analytics Vidhya, accessed May 4, 2025, [https://www.analyticsvidhya.com/blog/2024/10/chunking-techniques-to-build-exceptional-rag-systems/](https://www.analyticsvidhya.com/blog/2024/10/chunking-techniques-to-build-exceptional-rag-systems/)  
28. How to Chunk Documents for RAG - Multimodal.dev, accessed May 4, 2025, [https://www.multimodal.dev/post/how-to-chunk-documents-for-rag](https://www.multimodal.dev/post/how-to-chunk-documents-for-rag)  
29. Chunking Strategies for LLM Applications - Pinecone, accessed May 4, 2025, [https://www.pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/)  
30. Improving RAG Performance: WTF is Semantic Chunking? - Fuzzy ..., accessed May 4, 2025, [https://www.fuzzylabs.ai/blog-post/improving-rag-performance-semantic-chunking](https://www.fuzzylabs.ai/blog-post/improving-rag-performance-semantic-chunking)  
31. Chunking methods in RAG: comparison - BitPeak, accessed May 4, 2025, [https://bitpeak.com/chunking-methods-in-rag-methods-comparison/](https://bitpeak.com/chunking-methods-in-rag-methods-comparison/)  
32. How Retrieval Augmented Generation (RAG) Makes LLM Smarter - AltexSoft, accessed May 4, 2025, [https://www.altexsoft.com/blog/retrieval-augmented-generation-rag/](https://www.altexsoft.com/blog/retrieval-augmented-generation-rag/)  
33. Advanced Chunking Techniques for Better RAG Performance - Chitika, accessed May 4, 2025, [https://www.chitika.com/advanced-chunking-techniques-rag/](https://www.chitika.com/advanced-chunking-techniques-rag/)  
34. Semantic Chunking | VectorHub by Superlinked, accessed May 4, 2025, [https://superlinked.com/vectorhub/articles/semantic-chunking](https://superlinked.com/vectorhub/articles/semantic-chunking)  
35. How we are doing RAG AI evaluation in Atlas - ClearPeople, accessed May 4, 2025, [https://www.clearpeople.com/blog/how-we-are-doing-rag-ai-evaluation-in-atlas](https://www.clearpeople.com/blog/how-we-are-doing-rag-ai-evaluation-in-atlas)  
36. RAG systems: Best practices to master evaluation for accurate and reliable AI. | Google Cloud Blog, accessed May 4, 2025, [https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)  
37. How to Measure RAG from Accuracy to Relevance? - - Datategy, accessed May 4, 2025, [https://www.datategy.net/2024/09/27/how-to-measure-rag-from-accuracy-to-relevance/](https://www.datategy.net/2024/09/27/how-to-measure-rag-from-accuracy-to-relevance/)  
38. \[2309.15217\] Ragas: Automated Evaluation of Retrieval Augmented Generation - arXiv, accessed May 4, 2025, [https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)  
39. arXiv:2309.15217v1 \[cs.CL\] 26 Sep 2023, accessed May 4, 2025, [https://arxiv.org/pdf/2309.15217](https://arxiv.org/pdf/2309.15217)  
40. Community - Arxiv Dives - Oxen.ai, accessed May 4, 2025, [https://www.oxen.ai/community/arxiv-dives](https://www.oxen.ai/community/arxiv-dives)  
41. Evaluate RAG pipeline using Ragas in ```python with watsonx - IBM, accessed May 4, 2025, [https://www.ibm.com/think/tutorials/ragas-rag-evaluation-```python-watsonx](https://www.ibm.com/think/tutorials/ragas-rag-evaluation-```python-watsonx)  
42. Build a Retrieval Augmented Generation (RAG) App: Part 1 ..., accessed May 4, 2025, [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)  
43. An evaluation of RAG Retrieval Chunking Methods | VectorHub by ..., accessed May 4, 2025, [https://superlinked.com/vectorhub/articles/evaluation-rag-retrieval-chunking-methods](https://superlinked.com/vectorhub/articles/evaluation-rag-retrieval-chunking-methods)  
44. Advanced RAG on Hugging Face documentation using LangChain - Hugging Face Open-Source AI Cookbook, accessed May 4, 2025, [https://huggingface.co/learn/cookbook/advanced\_rag](https://huggingface.co/learn/cookbook/advanced_rag)  
45. RecursiveCharacterTextSplitter — LangChain documentation, accessed May 4, 2025, [https://python.langchain.com/api\_reference/text\_splitters/character/langchain\_text\_splitters.character.RecursiveCharacterTextSplitter.html](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)  
46. Text Splitters | 🦜️ LangChain, accessed May 4, 2025, [https://python.langchain.com/v0.1/docs/modules/data\_connection/document\_transformers/](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)  
47. How to recursively split text by characters | 🦜️ LangChain, accessed May 4, 2025, [https://python.langchain.com/docs/how\_to/recursive\_text\_splitter/](https://python.langchain.com/docs/how_to/recursive_text_splitter/)  
48. RecursiveCharacterTextSplitter | LangChain.js, accessed May 4, 2025, [https://v03.api.js.langchain.com/classes/langchain.text\_splitter.RecursiveCharacterTextSplitter.html](https://v03.api.js.langchain.com/classes/langchain.text_splitter.RecursiveCharacterTextSplitter.html)  
49. Self-Corrective RAG with LangGraph - Agentic RAG Tutorial - YouTube, accessed May 4, 2025, [https://www.youtube.com/watch?v=uZoz3T3Z6-w](https://www.youtube.com/watch?v=uZoz3T3Z6-w)  
50. LangChain Expression Language (LCEL) | 🦜️ Langchain, accessed May 4, 2025, [https://js.langchain.com/docs/concepts/lcel/](https://js.langchain.com/docs/concepts/lcel/)  
51. Evaluating RAG Applications with RAGAs - Towards Data Science, accessed May 4, 2025, [https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a/](https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a/)  
52. Extrinsic Hallucinations in LLMs | Lil'Log, accessed May 4, 2025, [https://lilianweng.github.io/posts/2024-07-07-hallucination/](https://lilianweng.github.io/posts/2024-07-07-hallucination/)  
53. Grounding LLMs to In-prompt Instructions: Reducing Hallucinations Caused by Static Pre-training Knowledge - ACL Anthology, accessed May 4, 2025, [https://aclanthology.org/2024.safety4convai-1.1.pdf](https://aclanthology.org/2024.safety4convai-1.1.pdf)  
54. A benchmark for evaluating conversational RAG - IBM Research, accessed May 4, 2025, [https://research.ibm.com/blog/conversational-RAG-benchmark](https://research.ibm.com/blog/conversational-RAG-benchmark)  
55. Ragas Writing Format - HackMD, accessed May 4, 2025, [https://hackmd.io/@KSLab-M1/BJyoUe3nC](https://hackmd.io/@KSLab-M1/BJyoUe3nC)  
56. Faithfulness - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/stable/concepts/metrics/available\_metrics/faithfulness/](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)  
57. Get better RAG responses with Ragas - Redis, accessed May 4, 2025, [https://redis.io/blog/get-better-rag-responses-with-ragas/](https://redis.io/blog/get-better-rag-responses-with-ragas/)  
58. Answer Relevance - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer\_relevance.html](https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_relevance.html)  
59. Context Precision - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/stable/concepts/metrics/available\_metrics/context\_precision/](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)  
60. Tutorial - Evaluate RAG Responses using Ragas | Couchbase Developer Portal, accessed May 4, 2025, [https://developer.couchbase.com/tutorial-evaluate-rag-responses-using-ragas/](https://developer.couchbase.com/tutorial-evaluate-rag-responses-using-ragas/)  
61. Metrics | Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/v0.1.21/references/metrics.html](https://docs.ragas.io/en/v0.1.21/references/metrics.html)  
62. Evaluating RAG Applications with RAGAs | Towards Data Science, accessed May 4, 2025, [https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a](https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a)  
63. Evaluate a simple RAG - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/stable/getstarted/rag\_eval/](https://docs.ragas.io/en/stable/getstarted/rag_eval/)  
64. Evaluating Using Your Test Set - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/v0.1.21/getstarted/evaluation.html](https://docs.ragas.io/en/v0.1.21/getstarted/evaluation.html)  
65. Observability Tools. - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/latest/howtos/observability/](https://docs.ragas.io/en/latest/howtos/observability/)  
66. Generating Synthetic Dataset for RAG - Prompt Engineering Guide, accessed May 4, 2025, [https://www.promptingguide.ai/applications/synthetic\_rag](https://www.promptingguide.ai/applications/synthetic_rag)  
67. Mastering Data: Generate Synthetic Data for RAG in Just $10 - Galileo AI, accessed May 4, 2025, [https://www.galileo.ai/blog/synthetic-data-rag](https://www.galileo.ai/blog/synthetic-data-rag)  
68. Ragas Synthetic Data Generation Methods | Restackio, accessed May 4, 2025, [https://www.restack.io/p/ragas-answer-synthetic-data-generation-methods-cat-ai](https://www.restack.io/p/ragas-answer-synthetic-data-generation-methods-cat-ai)  
69. Evaluate Using Metrics - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/v0.2.9/getstarted/rag\_evaluation/](https://docs.ragas.io/en/v0.2.9/getstarted/rag_evaluation/)  
70. Evaluate your first LLM App - Ragas, accessed May 4, 2025, [https://docs.ragas.io/en/latest/getstarted/evals/](https://docs.ragas.io/en/latest/getstarted/evals/)  
71. How do I improve RAG extracted document list : r/LangChain - Reddit, accessed May 4, 2025, [https://www.reddit.com/r/LangChain/comments/199ejhc/how\_do\_i\_improve\_rag\_extracted\_document\_list/](https://www.reddit.com/r/LangChain/comments/199ejhc/how_do_i_improve_rag_extracted_document_list/)  
72. Semantic Chunking for RAG: Better Context, Better Results - Multimodal.dev, accessed May 4, 2025, [https://www.multimodal.dev/post/semantic-chunking-for-rag](https://www.multimodal.dev/post/semantic-chunking-for-rag)  
73. Optimizing RAG with Advanced Chunking Techniques - Antematter, accessed May 4, 2025, [https://antematter.io/blogs/optimizing-rag-advanced-chunking-techniques-study](https://antematter.io/blogs/optimizing-rag-advanced-chunking-techniques-study)  
74. How to split text based on semantic similarity | 🦜️ LangChain, accessed May 4, 2025, [https://python.langchain.com/docs/how\_to/semantic-chunker/](https://python.langchain.com/docs/how_to/semantic-chunker/)  
75. Adding a max chunk size with SemanticChunker \#18014 - GitHub, accessed May 4, 2025, [https://github.com/langchain-ai/langchain/discussions/18014](https://github.com/langchain-ai/langchain/discussions/18014)  
76. SemanticChunker — LangChain documentation, accessed May 4, 2025, [https://api.python.langchain.com/en/latest/experimental/text\_splitter/langchain\_experimental.text\_splitter.SemanticChunker.html](https://api.python.langchain.com/en/latest/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)  
77. Langchain Semantic Chunker Overview - Restack, accessed May 4, 2025, [https://www.restack.io/docs/langchain-knowledge-semantic-chunker-cat-ai](https://www.restack.io/docs/langchain-knowledge-semantic-chunker-cat-ai)  
78. langchain\_experimental.text\_splitter.SemanticChunker - Langchain python API Reference, accessed May 4, 2025, [https://api.python.langchain.com/en/latest/text\_splitter/langchain\_experimental.text\_splitter.SemanticChunker.html](https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)  
79. Semantic Chunking Without OpenAI · langchain-ai langchain · Discussion \#17072 - GitHub, accessed May 4, 2025, [https://github.com/langchain-ai/langchain/discussions/17072](https://github.com/langchain-ai/langchain/discussions/17072)  
80. 8 Types of Chunking for RAG Systems - Analytics Vidhya, accessed May 4, 2025, [https://www.analyticsvidhya.com/blog/2025/02/types-of-chunking-for-rag-systems/](https://www.analyticsvidhya.com/blog/2025/02/types-of-chunking-for-rag-systems/)  
81. Benchmarking and Evaluating RAG - Part 1 - NeoITO Blog, accessed May 4, 2025, [https://www.neoito.com/blog/benchmarking-and-evaluating-rag-part-1/](https://www.neoito.com/blog/benchmarking-and-evaluating-rag-part-1/)