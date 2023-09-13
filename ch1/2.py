from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)
question = "Which NFL team won the Super Bowl in the 2010 season?"
davinci = OpenAI(model_name='text-davinci-003', openai_api_key=open('../openai-api-key.txt').read())
llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.run(question))

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
res = llm_chain.generate(qs)
print(res)
