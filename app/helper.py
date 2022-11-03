import app.shared_context as sc
from redis.commands.search.field import VectorField, TextField


def create_flat_index(vector_field_name, number_of_vectors, index_name="idx_txt"):
    fields = [
        VectorField(
            vector_field_name,
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": sc.TEXT_EMBEDDING_DIMENSION,
                "DISTANCE_METRIC": sc.TEXT_DISTANCE_METRIC,
                "INITIAL_CAP": number_of_vectors,
                "BLOCK_SIZE": number_of_vectors,
            }
        ),
        TextField("id"),
    ]
    if index_name == "idx_txt":
        fields.append(TextField("sentence"))
    sc.api_redis_cli.ft(index_name=index_name).create_index(fields)
