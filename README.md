Following book `LangChain: Introduction and Getting Started` at:
https://www.pinecone.io/learn/series/langchain/langchain-intro/

Focusing on OpenAPI

Ch1 - 5

Got OpenAPI API key from https://platform.openai.com/account/api-keys

## TODO

- [x] ch1: An Introduction to LangChain
    * Query an LLM via LLMChain with prompt via PromptTemplate
- [x] ch2: Prompt Templates and the Art of Prompts
    * Provide examples to LLM via LLMChain with prompt via FewShotPromptTemplate
    * Control amount of example text passed to LLM via FewShotPromptTemplate with example_selector via LengthBasedExampleSelector
- [x] ch3: Building Composable Pipelines with Chains
    * 3 types of chains
        * LLMChain: Send prompt to LLM and get result
        * TransformChain: Function that returns a value for each specified output variable for each specified input variable
        * SequentialChain: List of chains to execute, given each specified input and output variable
- [x] ch4: Conversational Memory for LLMs with Langchain
    * LLMs are stateless
    * Problem: Cannot have a conversation with an LLM.
    * Solution: Spoof state by passing conversation history before next query
    * ConversationChain: Extends LLMChain. Adds memory parameter via:
        * ConversationBufferMemory: All text up to this point.
        * ConversationSummaryMemory: Use LLM to summarize the current { history, query, and response } for use as a history in the next query.
        * ConversationBufferWindowMemory: Like ConversationBufferMemory, but only remember the last k interactions {query + response)
        * ConversationSummaryBufferMemory: Like ConversationSummaryMemory, but only remember the last k tokens
        * Other types exist
- [ ] ch5: Retrieval augmentation
    * Parametric knowledge: Data the LLM was trained on originally
    * Source knowledge: Data passed to LLM via prompt
    * Convert text data into chunks (a chunk is a list of tokens) via RecursiveCharacterTextSplitter
    * Convert chunks into numerical representations that are readable by an LLM