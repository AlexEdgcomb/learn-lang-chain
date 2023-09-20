import tiktoken

tokenizer = tiktoken.get_encoding('p50k_base')


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


length = tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
             "we can find the length of this chunk of text in tokens")

print(length)