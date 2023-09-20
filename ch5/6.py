from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from tqdm.auto import tqdm
from uuid import uuid4

# Source knowledge
data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')

# Chunker
tokenizer = tiktoken.get_encoding('p50k_base')


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# Vectorizer
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    document_model_name=model_name,
    query_model_name=model_name,
    openai_api_key=open('../openai-api-key.txt').read()
)

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embed.embed_documents(texts)

# Vector database
index_name = 'langchain-retrieval-augmentation'

pinecone.init(
    api_key=open('../pinecone-api-key.txt').read(),
    environment=open('../pinecone-environment.txt').read()
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=len(res[0])  # 1536 dim of text-embedding-ada-002
    )

index = pinecone.GRPCIndex(index_name)

# Indexer
batch_limit = 100

texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):
    # first get metadata fields for this record
    metadata = {
        'wiki-id': str(record['id']),
        'source': record['url'],
        'title': record['title']
    }
    # now we create chunks from the record text
    record_texts = text_splitter.split_text(record['text'])
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

print(index.describe_index_stats())
