import app.shared_context as sc
from app.helper import create_index, timeit
from redis.commands.search.query import Query


@timeit
def delete(pattern="txt:*"):
    keys = sc.api_redis_cli.keys(pattern)
    for key in keys:
        sc.api_logger.info(f"dropping key {key}")
        sc.api_redis_cli.delete(key)
    sc.api_logger.info(f"{len(keys)} keys dropped")


@timeit
def run(pattern="txt:*"):
    keys = sc.api_redis_cli.keys(pattern)
    bucket_size = [0, 0, 0, 0, 0]
    for key in keys:
        new_key = key[4:].decode()
        bucket = int(new_key) % sc.BUCKETS
        bucket_size[bucket] += 1
        obj = sc.api_redis_cli.hgetall(key)
        sc.api_redis_cli.hset(
            f"txt:{bucket}:{new_key}",
            mapping={
                "embedding": obj[b"embedding"],
                "sentence": obj[b"sentence"],
                "id": key
            }
        )
    # bucket_size = [200, 203, 193, 194, 210]
    sc.api_logger.info(f"bucket size: {bucket_size}")
    for bucket in range(sc.BUCKETS):
        create_index(
            index_name=f"idx_txt_{bucket}",
            distance_metric=sc.TEXT_DISTANCE_METRIC,
            vector_field_name="embedding",
            embedding_dimension=sc.TEXT_EMBEDDING_DIMENSION,
            number_of_vectors=bucket_size[bucket],
            index_type="HNSW",
            prefix=f"txt:{bucket}:"
        )


@timeit
def query(kws="iPhone", k=20):
    query_vector = sc.load_txt_model().encode(kws).astype(sc.TEXT_EMBEDDING_TYPE).tobytes()
    # FLAT - 1k docs
    q = Query(f"*=>[KNN {k} @embedding $query_vector AS score]")\
        .sort_by("score", asc=True)\
        .return_fields("id", "sentence", "score")\
        .dialect(2)
    params_dict = {"query_vector": query_vector}
    results_flat = sc.api_redis_cli.ft(index_name="idx_txt").search(q, query_params=params_dict)
    # HNSW - 5 buckets - ~200 docs each
    bucket_results = [[], [], [], [], []]
    for bucket in range(sc.BUCKETS):
        q = Query(f"*=>[KNN {k // sc.BUCKETS} @embedding $query_vector AS score]")\
            .sort_by("score", asc=True)\
            .return_fields("id", "sentence", "score")\
            .dialect(2)
        params_dict = {"query_vector": query_vector}
        results = sc.api_redis_cli.ft(index_name=f"idx_txt_{bucket}").search(q, query_params=params_dict)
        bucket_results[bucket] = list(results.docs)
    bucket_results.sort(key=lambda e: e.score)


if __name__ == '__main__':
    sc.api_redis_cli = sc.start_queueing()
    sc.api_logger = sc.start_encoder_logging()
    run()
