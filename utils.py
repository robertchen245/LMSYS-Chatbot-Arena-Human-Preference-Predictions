from transformers import GemmaTokenizerFast, PreTrainedTokenizerBase
import ast
from transformers import EvalPrediction
import torch
from sklearn.metrics import log_loss,accuracy_score
class CustomTokenizer():
    pass
class CustomTokenizerForHumanPreference(CustomTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self,batch:dict) -> dict:
        prompt = [self.process_text(t,"prompt") for t in batch["prompt"]]
        response_a = [self.process_text(t,'response_a') for t in batch["response_a"]]
        response_b = [self.process_text(t,'response_b') for t in batch["response_b"]]
        texts = [p + r_a + r_b for p,r_a,r_b in zip(prompt,response_a,response_b)]
        tokenized = self.tokenizer(texts,max_length=self.max_length,truncation=True)
        labels=[]
        for a_win, b_win in zip(batch["winner_model_a"],batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)
        return {**tokenized,"labels":labels}
    @staticmethod
    def process_text(text:str,prefix=str)->str:
        try:
            list_of_text = eval(text,{"null":""})
            return " ".join([f"<{prefix}:{i+1}>:" + text for i,text in enumerate(list_of_text)])
        except:
            print(text)
def compute_metrics(eval_pred:EvalPrediction): # Type: numpy NDArray
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels,y_pred=probs)
    acc = accuracy_score(y_true=labels,y_pred=preds.argmax(-1))
    return {"log_loss":loss,"acc":acc}