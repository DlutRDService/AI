from transformers import AutoTokenizer, BertModel
from app.db.milvus_conf import MilvusConf
from datasets import load_dataset
from typing import List, Dict


def embedding(text: str) -> List:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.pooler_output.tolist()[0]


def save_to_db(qa: Dict) -> None:
    for i in qa:
        i['id'] = int(i['id'])
        i["vector"] = embedding(i["question"])
        i['human_answers'] = str(i['human_answers'])
        i["chatgpt_answers"] = str(i["chatgpt_answers"])
        try:
            milvus_client.client.insert(
                collection_name="demo",
                data=i
            )
            print(f"成功导入问题:{i['question']}")
        except Exception as error:
            print(error)


if __name__ == '__main__':
    milvus_client = MilvusConf()

    ds = load_dataset("Hello-SimpleAI/HC3-Chinese", "finance", split="train")

    save_to_db(ds)
