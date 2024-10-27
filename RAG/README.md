# Empathetic WhatsApp Message Generation Using RAG

This project explores the use of **Retrieval-Augmented Generation (RAG)** to generate empathetic WhatsApp messages. On this section We compare three distinct datasets and two embedding methods to identify the optimal combination for enhancing our pipeline's performance in generating messages with emotional intelligence.

## Project Overview

The core mission of this project is to build a system capable of generating empathetic responses in WhatsApp conversations. To achieve this, we analyze on this section the performance of different embeddings on a series of queries across three datasets and two embedding models.

### Datasets

We use three diverse datasets to assess how well each embedding method captures emotional and relational nuances:

1. **Google_Emotions**: A dataset focused on various emotional states and expressions.
2. **Reddit_Relationship_Advice**: A dataset sourced from Reddit, containing real-life relationship advice discussions.
3. **Friends TV Show Scripts**: All the dialogue from the *Friends* TV show, capturing a broad range of interpersonal interactions.

### Embedding Methods

We apply two embedding methods to these datasets:

1. **microsoft/deberta-base**: A transformer-based model designed for improved handling of language understanding tasks, especially in emotional contexts.
2. **all-MiniLM-L6-v2**: A compact variation of the BERT model, designed for efficient performance while maintaining accuracy in semantic tasks.

### Query Comparison

We perform comparisons using five queries, running them across six variations of dataset-embedding combinations. This process allows us to examine which combination provides the most empathetic responses and the best contextual understanding.

The six variations are as follows:

1. Google_Emotions with microsoft/deberta-base
2. Google_Emotions with all-MiniLM-L6-v2
3. Reddit_Relationship_Advice with microsoft/deberta-base
4. Reddit_Relationship_Advice with all-MiniLM-L6-v2
5. Friends TV Show Scripts with microsoft/deberta-base
6. Friends TV Show Scripts with all-MiniLM-L6-v2

## Files Explanation

- **Dataset_Embeddings.py**: This script is responsible for creating embeddings for all the experimental datasets. It ensures the embeddings are generated using the selected models (microsoft/deberta-base and all-MiniLM-L6-v2) so they can be used in further comparisons.

- **Experimental_prompts**: A file containing the prompts we used to test the RAG pipeline. These prompts are designed to simulate real-world WhatsApp message requests and serve as the basis for evaluating the model's ability to generate empathetic responses.

- **RAG_Pipeline.ipynb**: This Jupyter notebook contains the core code where we examine the best source knowledge (datasets) and embedding methods. It also includes the final results after selecting the best combination for generating empathetic WhatsApp messages.

- **Queries_data_sets**: This file contains two sets of queries:
  1. The five queries used during the initial experimental phase to test different embedding and dataset combinations.
  2. The 200 sensitive queries used for the final evaluation between our RAG model, GPT-4, and Claude 3.5.




## Results

Through this experimentation, we aimed to identify the best combination of embedding method and dataset. Ultimately, the best results came from the **Friends TV Show Scripts** using **all-MiniLM-L6-v2**. This combination provided the most emotionally appropriate and contextually aware responses for generating empathetic WhatsApp messages.

You can observe the results of our initial tagging here- https://docs.google.com/spreadsheets/d/1pVycu26Fcn_w_dvVaVA4Qb4lg8V3SidFRZgczVpkM2k/edit?gid=0#gid=0
