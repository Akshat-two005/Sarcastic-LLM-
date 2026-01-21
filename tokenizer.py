import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

def encode(text):
    return sp.encode(text, out_type=int)

def decode(tokens):
    return sp.decode(tokens)
