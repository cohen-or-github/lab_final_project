# Empathetic Chatbot for WhatsApp Message Generation

This project focuses on building a system that generates empathetic WhatsApp messages for users dealing with unpleasant situations. We developed a chatbot using **Retrieval-Augmented Generation (RAG)** and compared its performance to state-of-the-art models like GPT-4 and Claude 3.5 to ensure the highest quality responses in terms of empathy, relevance, and clarity.

## Project Overview

The project aims to create a chatbot capable of generating emotionally sensitive and contextually appropriate WhatsApp messages. We followed a structured approach to build and optimize the chatbot:

1. **RAG Configuration Selection**: 
   - We examined three datasets: **Google_Emotions**, **Reddit_Relationship_Advice**, and the **Friends TV Show Scripts**.
   - Two embedding methods were tested: **microsoft/deberta-base** and **all-MiniLM-L6-v2**.
   - After running five queries across six combinations of datasets and embeddings, we determined that **Friends TV Show Scripts** with **all-MiniLM-L6-v2** was the best configuration for our pipeline.

2. **Model Comparison**:
   - We compared 200 sensitive WhatsApp message queries between our optimized RAG model, **GPT-4**, and **Claude 3.5**.
   - Each model was evaluated on the following three metrics:
     1. **Human-like Rate**: How natural and human-like the responses are.
     2. **Response Relevance Rate**: How well the responses match the context of the query.
     3. **Clarity & Coherence**: How clear and coherent the responses are.
   
   See [annotation guidelines](#annotation-guidelines) for further explanation of these metrics.

3. **Chatbot Implementation**:
   - The final RAG model, based on the **Friends TV Show Scripts**, was implemented into a chatbot.
   - Users can interact with the chatbot, asking for advice or messages tailored to unpleasant situations.

## How to Run Source Code

1. **Download the Cleaned Preprocessed Datasets**:
   - Ensure the cleaned datasets are available in the `/preprocess/clean_datasets` directory. These datasets will be used to create embeddings for the RAG pipeline.

2. **Generate Embeddings for the Datasets**:
   - Run the `Dataset_Embeddings.py` script on the `clean_datasets` directory. This script will create embeddings based on the experimental datasets, using the specified embedding methods (either microsoft/deberta-base or all-MiniLM-L6-v2).

3. **Run the RAG Pipeline**:
   - Open and execute the `RAG_Pipeline.ipynb` notebook.
   - This notebook allows you to:
     - Evaluate the performance of each dataset and embedding combination.
     - Run the final selected configuration as a knowledge base for generating responses to the 200 test queries.

4. **API Keys and Index Name Configuration**:
   - Before running `RAG_Pipeline.ipynb`, make sure to replace the placeholders for `COHERE_API_KEY` and `PINECONE_API_KEY` with your own keys.
   - Set `INDEX_NAME` according to the relevant dimensions:
     - Use `384` for embeddings generated with **all-MiniLM-L6-v2**.
     - Use `768` for embeddings generated with **microsoft/deberta-base**.

## How do run chat-bot via Streamlit app
   1. **API Keys and Index Name Configuration**:
      - Before running the app, make sure to replace the placeholders for `pinecone_api_key.txt` and `cohere_api_key.txt` in lab_final_project/streamlit/ with your own keys.
      - Set `INDEX_NAME` according to the relevant dimensions:
        - Use `384` for embeddings generated with **all-MiniLM-L6-v2**.

   2. **Streamlit Installation**
       Make sure you have installed Streamlit dependencies (pip install streamlit) and SentenceTransformer.
       
   2. **Run App**
      Navigate to the folder the streamlit app files are located in, and run the following command in the terminal: "streamlit run streamlit_akward_app.py"
      This will run the app on a local server.

   4. Open the local server link printed in your terminal in your browser. Enjoy!

   
![image](https://github.com/user-attachments/assets/54609f53-689f-4edd-bc54-48b9890206ca)

