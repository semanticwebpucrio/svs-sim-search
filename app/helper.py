import app.shared_context as sc
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


def create_index(vector_field_name,
                 number_of_vectors,
                 embedding_dimension,
                 distance_metric,
                 index_type="FLAT",
                 index_name="idx_txt",
                 prefix="*"
                 ):
    fields = [
        VectorField(
            vector_field_name,
            index_type,
            {
                "TYPE": "FLOAT32",
                "DIM": embedding_dimension,
                "DISTANCE_METRIC": distance_metric,
                "INITIAL_CAP": number_of_vectors,
            }
        ),
        TextField("id"),
    ]
    if "txt" in index_name:
        fields.append(TextField("sentence"))
    sc.api_redis_cli.ft(index_name=index_name).create_index(
        fields, definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
    )
