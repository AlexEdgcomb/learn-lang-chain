from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

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


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


queries = [
    'My interest here is to explore the potential of integrating Large Language Models with external knowledge',
    'I just want to analyze the different possibilities. What can you think of?',
    'Which data source types could be used to give context to the model?',
    'What is my aim again?'
]

for query in queries:
    print(count_tokens(conversation, query))
    print()

print(conversation.memory.buffer)
