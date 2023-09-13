from langchain.llms import OpenAI

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=open('../openai-api-key.txt').read()
)

prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is always sarcastic and witty. Here are some
examples:

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

openai.temperature = 1.0  # increase creativity/randomness of output

print(openai(prompt))
