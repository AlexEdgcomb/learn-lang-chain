from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

davinci = OpenAI(model_name='text-davinci-003', openai_api_key=open('../openai-api-key.txt').read())

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=davinci
)

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))
