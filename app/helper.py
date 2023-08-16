import time
from functools import wraps
import app.shared_context as sc
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
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


def slice_dataframe(dff, qtd):
    num_lines = dff.shape[0]
    lines_per_df = num_lines // qtd
    dfs = []
    ctrl = 0
    for _ in range(qtd - 1):
        dfs.append(dff.iloc[ctrl:ctrl+lines_per_df])
        ctrl += lines_per_df
    dfs.append(dff.iloc[ctrl:])
    return dfs
