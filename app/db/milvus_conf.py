from pymilvus import MilvusClient
from typing import List


class MilvusConf:
    def __init__(self) -> None:
        self.schema = None
        self.client = MilvusClient(
            uri="http://localhost:19530"
        )
        self.index_params = self.client.prepare_index_params()

    def create_schema(self, schema_fields: List[dict]):
        self.schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        for i in schema_fields:
            self.schema.add_field(
                field_name = i['name'],
                max_length = i['length'],
                datatype = i['type'],
                is_primary = i['is_primary'],
                dim = i['dim']
            )

    def create_collection(self, collection_name, schema):
        if schema is None:
            schema = self.schema
        self.client.create_collection(
            collection_name = collection_name,
            schema = schema
        )

    def drop_collection(self, collection_name):
        self.client.drop_collection(collection_name)

    def create_index(self):
        self.index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index",
            params={"nlist": 128}
        )

        self.client.create_index(
            collection_name="demo",
            index_params=self.index_params
        )

    def collection(self, collection_name):
        return self.client.load_collection(
            collection_name=collection_name
        )

    def sereach(self, sereach_vector: List, collection_name: str):
        res = self.client.search(
            collection_name=collection_name,
            data=sereach_vector,
            limit=2,
            search_params={
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
        )
        result = self.client.get(
            collection_name=collection_name,
            ids=res[0][0]['id']
        )
        return result


