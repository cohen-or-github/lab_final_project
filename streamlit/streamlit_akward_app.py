import streamlit as st
import pandas as pd
import numpy as np
import ast
from datasets import Dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import cohere
import time
from sentence_transformers import SentenceTransformer

with open('cohere_api_key.txt') as f:
    COHERE_API_KEY = f.read().strip()
with open("pinecone_api_key.txt") as f:
    PINECONE_API_KEY = f.read().strip()

# Initialize Cohere and Pinecone clients
co = cohere.Client(api_key=COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load and cache the Sentence Transformer model
@st.cache(allow_output_mutation=True)
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)


model = load_model()

# Load precomputed embeddings from CSV
@st.cache(allow_output_mutation=True)
def load_preembedded_data():
    LOCAL_CSV_PATH = "Friends_Transcript_Preprocessed_with_Embeddings_Cleaned.csv"
    df = pd.read_csv(LOCAL_CSV_PATH)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    dataset = Dataset.from_pandas(df)
    embeddings = np.stack(df['embedding'].values)
    return dataset, embeddings
dataset, embeddings = load_preembedded_data()


# Create or load a Pinecone index
def create_pinecone_index(index_name: str, dimension: int, metric: str = 'cosine'):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


index = pc.Index("mini-lm-6")

# Upsert vectors into Pinecone
def upsert_vectors(index, embeddings, dataset, text_field='full_text_with_label', batch_size=128):
    ids = [str(i) for i in range(len(embeddings))]
    meta = [{text_field: text} for text in dataset[text_field][:len(embeddings)]]
    to_upsert = [{"id": ids[i], "values": embeddings[i].tolist(), "metadata": meta[i]} for i in range(len(ids))]
    for i in tqdm(range(0, len(to_upsert), batch_size)):
        index.upsert(vectors=to_upsert[i:i + batch_size])
    return index


if "index_upserted" not in st.session_state:
    upsert_vectors(index, embeddings, dataset)
    st.session_state["index_upserted"] = True


# Define prompt augmentation function
def augment_prompt(query: str, model: SentenceTransformer, index) -> str:
    query_vector = model.encode(query)
    query_results = index.query(
        vector=query_vector.tolist(),
        top_k=3,
        include_values=True,
        include_metadata=True
    )['matches']
    text_matches = [match['metadata'].get('full_text_with_label', 'No content found') for match in query_results]
    source_knowledge = "\n\n".join(text_matches)
    augmented_prompt = f"""You are tasked with composing a WhatsApp message that you would send directly to the person, maintaining the tone, style, and level of empathy and directness used in the provided source material.

    The response should simulate a real-time, casual WhatsApp message.
    Ensure the tone is empathetic and conversational, while remaining concise and clear.
    Use the human writing style from the source knowledge as a guide, but note that the source knowledge does not contain direct answers to the query
    
    Important Guidelines:
    The response must directly answer the query as if you are sending the message right now.
    Maintain the casual tone, while ensuring the message is smooth and empathetic, like a typical WhatsApp conversation.
    The source knowledge is provided solely to show the desired tone and writing style. It is not to be used as a source of answers or content for the response.
    Stick strictly to the format of a direct message, avoiding extra advice or unwarranted sympathy.
    
    
    Example Query and Response Format:
    
    Query: “I agreed to be a bridesmaid, but now I can’t commit. How can I let the bride know without causing drama?”
    
    Response: “Hey, I don’t know if you’ve noticed, but I’m kind of freaking out about this whole bridesmaid thing. I’m so sorry, but I don’t think I can do it anymore. I know how important your wedding is, and I don’t want to let you down, but I’m just not in the right headspace. I hope you understand, and that we can still be cool.”
    The source knowledge is as follows, and it is highly important to use the context and the human writing style in it to write the message:
    {source_knowledge}
    Query: {query}"""

    return augmented_prompt, source_knowledge


st.markdown(
    """
    <style>
    /* Set the background color */
    body, .stApp {
        background-color: #ffcccc;
    }
    /* Center container with a narrower width for the text boxes */
    .container {
        max-width: 600px;
        margin: auto;
    }
    /* Center-align the title */
    .title {
        text-align: center;
    }
    /* Style input and output text areas to have the same width */
    .container .stTextArea textarea {
        width: 100% !important;
        background-color: #f0f0f0;
        color: #333;
        padding: 10px;
        border-radius: 5px;
    }
    /* Center the button and reduce space between text area and button */
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px; /* Add small gap to control spacing */
    }
    .stButton button {
        background-color: #ff8888;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 20px;
    }
    /* Remove extra padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Columns layout: Image on the left, content on the right
col1, col2 = st.columns([1, 2])

# Display the images in the first column with a top margin
with col1:
    st.markdown('<div class="lowered-image">', unsafe_allow_html=True)
    st.image("girl.jpg", use_column_width=True)  # Replace "girl.jpg" with the path to your image
    st.image("guy.jpg", use_column_width=True)  # Replace "guy.jpg" with the path to your second image
    st.markdown('</div>', unsafe_allow_html=True)

# Main content in the second column
with col2:
    st.markdown("<h3 class='title'>What's the awkward message of the day? 😳</h3>", unsafe_allow_html=True)
    query3 = st.text_area("", value="")

    # Center the button and spinner within the form container
    if st.button("Phrase it!"):
        with st.spinner("Phrasing..."):
            time.sleep(2)
            augmented_prompt, source_knowledge = augment_prompt(query3, model, index)
            response = co.chat(model='command-r-plus', message=augmented_prompt)
            response_text = response.text
            st.text_area("", value=response_text, height=200)