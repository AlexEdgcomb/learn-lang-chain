from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

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

# Query vector store
text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# completion llm
llm = ChatOpenAI(
    openai_api_key=open('../openai-api-key.txt').read(),
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "who was Benito Mussolini?"

print(qa_with_sources(query))
