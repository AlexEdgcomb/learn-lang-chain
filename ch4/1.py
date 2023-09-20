from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key=open('../openai-api-key.txt').read(),
	model_name="text-davinci-003"
)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)

print(conversation.prompt.template)
