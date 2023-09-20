from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key=open('../openai-api-key.txt').read(),
	model_name="text-davinci-003"
)

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

print(conversation("Good morning AI!"))
