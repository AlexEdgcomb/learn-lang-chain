from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from datasets import load_dataset

data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
print('Source text')
print(data[6])

tokenizer = tiktoken.get_encoding('p50k_base')


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(data[6]['text'])[:3]
print()
print('Chunks')
print(chunks)

print()
print('Size of each chunk', (tiktoken_len(chunks[0]), tiktoken_len(chunks[1]), tiktoken_len(chunks[2])))
