import pandas as pd
from sentence_transformers import SentenceTransformer


# Function to encode text in batches and handle data as lists
def encode_text_in_batches(text_series,model,batch_size=32):
    text_list = text_series.tolist()  # Convert Series to list
    batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
    all_embeddings = []
    for batch in batches:
        embeddings = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
        all_embeddings.extend(embeddings)
    return all_embeddings


data_set_lists =['Friends_Preprocessed.csv','Google_Emotions_Preprocessed.csv','Reddit_Relationship_Advice_Preprocessed.csv']
EMBEDDING_MODELS = ['all-MiniLM-L6-v2','microsoft/deberta-base']


# Generate embeddings for the 'full_text_with_label' column
for data_set in data_set_lists:
    df = pd.read_csv(data_set)
    for EMBEDDING_MODEL in EMBEDDING_MODELS:
        model = SentenceTransformer(EMBEDDING_MODEL)
        df['embedding'] = encode_text_in_batches(df['full_text'],model)
        if EMBEDDING_MODEL == 'microsoft/deberta-base':
            df.to_csv(f'deberta_{data_set}', index=False)
        else:
            df.to_csv(f'bert_{data_set}', index=False)

