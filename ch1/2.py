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
