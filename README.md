Following book `LangChain: Introduction and Getting Started` at:
https://www.pinecone.io/learn/series/langchain/langchain-intro/

Focusing on OpenAPI

Ch1 - 5

Got OpenAPI API key from https://platform.openai.com/account/api-keys

## TODO
- [x] ch1
    * Query an LLM via LLMChain with prompt via PromptTemplate
- [x] ch2
    * Provide examples to LLM via LLMChain with prompt via FewShotPromptTemplate
    * Control amount of example text passed to LLM via FewShotPromptTemplate with example_selector via LengthBasedExampleSelector
- [x] ch3
    * 3 types of chains
        * LLMChain: Send prompt to LLM and get result
        * TransformChain: Function that returns a value for each specified output variable for each specified input variable
        * SequentialChain: List of chains to execute, given each specified input and output variable
- [ ] ch4
- [ ] ch5