import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

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

index_name = 'langchain-retrieval-augmentation'

pinecone.init(
    api_key=open('../pinecone-api-key.txt').read(),
    environment=open('../pinecone-environment.txt').read()
)

# we create a new index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=len(res[0])  # 1536 dim of text-embedding-ada-002
    )

index = pinecone.GRPCIndex(index_name)

print(index.describe_index_stats())
