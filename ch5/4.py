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
print(len(res), len(res[0]))
