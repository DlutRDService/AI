from torch import tensor
from transformers import AutoTokenizer, BertModel
from datasets import load_dataset
from typing import Dict
from app.db.milvus_conf import MilvusConf





# save_to_db(ds)


# input = f"{query} 以下是相关人员给出的部分解答，理解这些解答并给出你的回复。"

# index_params = MilvusClient.prepare_index_params()
#
# # 4.2. Add an index on the vector field.
# index_params.add_index(
#     field_name="vector",
#     metric_type="COSINE",
#     index_type="IVF_FLAT",
#     index_name="vector_index",
#     params={"nlist": 128}
# )
#
# # 4.3. Create an index file
# client.create_index(
#     collection_name="demo",
#     index_params=index_params
# )
#
# schema = MilvusClient.create_schema(
#     auto_id=False,
#     enable_dynamic_field=True,
# )
#
# schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
# schema.add_field(field_name="question", max_length=4096, datatype=DataType.VARCHAR, is_primary=False)
# schema.add_field(field_name="human_answers", max_length=4096, datatype=DataType.VARCHAR, is_primary=False)
# schema.add_field(field_name="chatgpt_answers", max_length=4096, datatype=DataType.VARCHAR, is_primary=False)
# schema.add_field(field_name="vector", max_length=4096, datatype=DataType.FLOAT_VECTOR, dim=768)
# client = MilvusClient(
#     uri="http://localhost:19530"
# )
# client.drop_collection(
#     collection_name="demo"
# )
# client.create_collection(
#     collection_name="demo",
#     schema=schema,
# )
