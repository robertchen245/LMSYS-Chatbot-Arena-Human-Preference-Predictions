from transformers import GemmaTokenizerFast
import matplotlib.pyplot as plt
import pandas as pd
import ast
def token_statistics(corpus: list, image_path: str = None, tokenizer=None):
    if tokenizer is None:
        return None
    all_tokens = []
    for i in range(len(corpus)):
        try:
            tokens = tokenizer(corpus[i])
        except:
            continue
        all_tokens.append(len(tokens["input_ids"]))
    if image_path is not None:
        plt.figure(figsize=(10, 5))
        plt.hist(
            all_tokens, bins=30, edgecolor="black"
        )  
        plt.title("Token Count Distribution")  
        plt.xlabel("Token Count")  
        plt.ylabel("Frequency")  
        plt.grid(True)  
        plt.savefig(image_path)
    return all_tokens
if __name__ == "__main__":
    tokenizer = GemmaTokenizerFast.from_pretrained("../gemma-2-9b-it-bnb-4bit")
    df = pd.read_csv("train.csv")
    corpus = []
    for i in range(len(df)):
        try:
            prompts = ast.literal_eval(df.loc[i, "prompt"])
            response_a = ast.literal_eval(df.loc[i, "response_a"])
            response_b = ast.literal_eval(df.loc[i, "response_b"])
            single_corpus = " ".join(prompts + response_a + response_b)
            corpus.append(single_corpus)
        except:
            continue
        if i % 3000 == 0 and i != 0:
            print(f"{i} row finished")
    token_statistics(corpus=corpus,image_path="statistics.png",tokenizer=tokenizer)
