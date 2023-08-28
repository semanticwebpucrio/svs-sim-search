import pdb
import json
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import app.shared_context as sc
from app.helper import create_index, timeit
from redis.commands.search.query import Query


@timeit
def to_redis(key_prefix="txt", file_name="txt_embeddings.parquet", batch_size=10_000, buckets=5):
    output_path = Path.cwd() / "app" / "output"
    parquet_file = pq.ParquetFile(output_path / file_name)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        print(f"loading batch of embedding - {batch_size}")
        df = batch.to_pandas()
        print("iterating over batch dataframe")
        for idx, row in df.iterrows():
            list_id = row["id"]
            keys = row.keys()
            sc.api_redis_cli.hset(
                f"{key_prefix}::{list_id}",
                mapping={k: row[k] for k in keys}
            )
            bucket = idx % buckets
            sc.api_redis_cli.hset(
                f"{key_prefix}:{bucket}:{list_id}",
                mapping={k: row[k] for k in keys}
            )


@timeit
def from_redis(pattern="txt::*", dest_file_name="txt_embeddings.parquet"):
    sc.api_logger.info(f"reading keys from Redis - pattern {pattern}")
    keys = sc.api_redis_cli.keys(pattern)
    dest = Path.cwd() / dest_file_name
    rows = []
    for idx, key in enumerate(keys):
        sc.api_logger.info(f"{idx} - extracting and converting key {key.decode()}")
        elem = sc.api_redis_cli.hgetall(key)
        elem = {key.decode(): val.decode() if key.decode() != "embedding" else val for key, val in elem.items()}
        rows.append(elem)
    sc.api_logger.info("generating Pandas dataframe")
    df = pd.DataFrame(rows)
    sc.api_logger.info("persisting Pandas dataframe as Parquet")
    df.to_parquet(dest)


@timeit
def rename(pattern="txt:*", prefix="txt::"):
    keys = sc.api_redis_cli.keys(pattern)
    for key in keys:
        if str(key).count(":") == 2:
            sc.api_logger.info(f"skipping key {str(key)}")
            continue
        # FIXME: this selection made on list must be dynamic
        new_key = int(key[4:].decode())
        obj = sc.api_redis_cli.hgetall(key)
        sc.api_redis_cli.hset(
            f"{prefix}{new_key}",
            mapping={
                "embedding": obj[b"embedding"],
                "sentence": obj[b"sentence"],
                "id": key
            }
        )
        sc.api_redis_cli.delete(key)


@timeit
def delete(pattern="txt:*"):
    keys = sc.api_redis_cli.keys(pattern)
    for key in keys:
        sc.api_logger.info(f"dropping key {key}")
        sc.api_redis_cli.delete(key)
    sc.api_logger.info(f"{len(keys)} keys dropped")


@timeit
def run(pattern="txt:*", only_index=False):
    if not only_index:
        print("Duplicando Dados para a Criacao dos Indices")
        keys = sc.api_redis_cli.keys(pattern)
        bucket_size = [0, 0, 0, 0, 0]
        for key in keys:
            new_key = key[5:].decode()
            bucket = int(new_key) % sc.BUCKETS
            bucket_size[bucket] += 1
            obj = sc.api_redis_cli.hgetall(key)
            sc.api_redis_cli.hset(
                f"txt:{bucket}:{new_key}",
                mapping={
                    "embedding": obj[b"embedding"],
                    # "sentence": obj[b"sentence"],
                    "id": new_key
                }
            )
    print("Criando Indice Geral")
    create_index(
          index_name="idx_txt",
          distance_metric=sc.TEXT_DISTANCE_METRIC,
          vector_field_name="embedding",
          embedding_dimension=sc.TEXT_EMBEDDING_DIMENSION,
          # number_of_vectors=50000,
          index_type="HNSW",
          prefix="txt::"
    )
    # bucket_size = [9907, 10103, 9869, 10053, 10068]
    sc.api_logger.info(f"bucket size: {bucket_size}")
    for bucket in range(sc.BUCKETS):
        print(f"Criando Indice {bucket}")
        create_index(
            index_name=f"idx_txt_{bucket}",
            distance_metric=sc.TEXT_DISTANCE_METRIC,
            vector_field_name="embedding",
            embedding_dimension=sc.TEXT_EMBEDDING_DIMENSION,
            # number_of_vectors=bucket_size[bucket],
            index_type="HNSW",
            prefix=f"txt:{bucket}"
        )


@timeit
def query_index_img(index_name, img_path, k):
    query_vector = sc.encode_image(img_path=img_path).astype(sc.IMG_EMBEDDING_TYPE).tobytes()
    q = Query(
            f"*=>[KNN {k} @embedding $query_vector AS score]"
        ).sort_by(
            "score", asc=False
        ).return_fields(
            "id", "score"
        ).dialect(2)
    params_dict = {"query_vector": query_vector}
    results = sc.api_redis_cli.ft(index_name=index_name).search(q, query_params=params_dict)
    return [(doc.id, doc.score) for doc in results.docs]  # key, score


@timeit
def query_index_txt(index_name, kws, k):
    query_vector = sc.load_txt_model().encode(kws).astype(sc.TEXT_EMBEDDING_TYPE).tobytes()
    q = Query(
            f"*=>[KNN {k} @embedding $query_vector AS score]"
        ).sort_by(
            "score", asc=False
        ).return_fields(
            "id", "score"
        ).dialect(2)
    params_dict = {"query_vector": query_vector}
    results = sc.api_redis_cli.ft(index_name=index_name).search(q, query_params=params_dict)
    return [(doc.id, doc.score) for doc in results.docs]  # key, score


@timeit
def query(k=20, buckets=sc.BUCKETS):
    input_path = Path.cwd() / "app" / "input"
    with open(input_path / "query_inputs.json", "r") as file:
        inputs = json.load(file)
    analysis = {}
    for idx, val in inputs.items():
        k_full = 10 if k > 10 else k
        k_bucket = k // buckets
        # txt
        result_flat_txt = query_index_txt("idx_txt_flat")
        result_hnsw_txt = query_index_txt("idx_txt")
        result_buckets_txt = [r for r in query_index_txt(f"idx_txt_{bucket}") for bucket in range(sc.BUCKETS)]
        breakpoint()
        # analysis[val["id"]] = {"flat": query_index("idx_txt_flat", val["caption"], k)}
        # analysis[row["id"]].update({"hnsw": query_index("idx_txt", row["caption"], k)})
        # bucket_results = []
        # for bucket in range(sc.BUCKETS):
        #     results = query_index(f"idx_txt_{bucket}", row["caption"], k // sc.BUCKETS)
        #     analysis[row["id"]].update({f"hnsw_bucket_{bucket}": results})
        #     bucket_results += list(results.docs)
        # bucket_results.sort(key=lambda e: float(e.score))
        # analysis[row["id"]].update({f"hnsw_buckets": bucket_results})
    return analysis


def calculate_metrics(results: dict):
    K = (1, 5, 10)
    queries = list(results.keys())
    for idx, q in enumerate(queries):
        print(f"Q{idx + 1} - {q}")
        flat = results[q]["flat"]["docs"]
        hnsw = results[q]["hnsw"]["docs"]
        hnswb = results[q]["hnsw_buckets"]
        rflat = [set([e["id"][5:] for e in flat[:i]]) for i in K]
        rhnsw = [set([e["id"][5:] for e in hnsw[:i]]) for i in K]
        rhnswb = [set([e["id"][6:] for e in hnswb[:i]]) for i in K]
        for i, k in enumerate(K):
            a = len(rflat[i] & rhnsw[i])
            c = len(rhnsw[i] - rflat[i])
            # b = len(rflat[i] - rhnsw[i])
            b = len(flat)
            print(f"HNSW precision@{k} - {a/(a+c)}")
            print(f"HNSW recall@{k} - {a/b}")
            a = len(rflat[i] & rhnswb[i])
            c = len(rhnswb[i] - rflat[i])
            # b = len(rflat[i] - rhnswb[i])
            b = len(flat)
            print(f"HNSW buckets precision@{k} - {a/(a+c)}")
            print(f"HNSW buckets recall@{k} - {a/b}")

@timeit
def create_buckets(pattern="txt::*", num_buckets=5):
    keys = sc.api_redis_cli.keys(pattern)
    output_path = Path.cwd() / "app" / "output"
    with open(output_path / f"create_buckets_{pattern[:3]}.error", "a") as fpe:
        for idx, key in enumerate(keys):
            elem = sc.api_redis_cli.hgetall(key)
            if len(elem.keys()) == 0:
                fpe.write("{list_id} => empty values")
                fpe.write("\n")
            bucket = idx % num_buckets
            sc.api_logger.info(f"{idx} - duplicating key {key.decode()} into bucket {bucket}")
            list_id = elem["id".encode()]
            sc.api_redis_cli.hset(
                f"txt:{bucket}:{list_id}",
                mapping={k: elem[k] for k in elem.keys()}
            )
    sc.api_logger.info(f"{len(keys)} keys added in {num_buckets} buckets")
    sc.api_logger.info(f"ERRORS: {len(errors)} - {errors}")


@timeit
def create_bucket_txt_index(bucket):
    idx_name = f"idx_txt_{bucket}"
    print(f"creating index {idx_name}")
    create_index(
        index_name=idx_name,
        distance_metric=sc.TEXT_DISTANCE_METRIC,
        vector_field_name="embedding",
        embedding_dimension=sc.TEXT_EMBEDDING_DIMENSION,
        index_type="HNSW",
        prefix=f"txt:{bucket}:"
    )


@timeit
def create_bucket_img_index(bucket):
    idx_name = f"idx_img_{bucket}"
    print(f"creating index {idx_name}")
    create_index(
        index_name=idx_name,
        distance_metric=sc.IMG_DISTANCE_METRIC,
        vector_field_name="embedding",
        embedding_dimension=sc.IMG_EMBEDDING_DIMENSION,
        index_type="HNSW",
        prefix=f"img:{bucket}:"
    )


@timeit
def create_full_txt_index(index_type):
    if index_type not in ["FLAT", "HNSW"]:
        raise IndexError("invalid index_type - must be FLAT or HNSW")
    idx_name = "idx_txt" if index_type == "HNSW" else "idx_txt_flat"
    print(f"creating index {idx_name}")
    create_index(
        index_name=idx_name,
        distance_metric=sc.TEXT_DISTANCE_METRIC,
        vector_field_name="embedding",
        embedding_dimension=sc.TEXT_EMBEDDING_DIMENSION,
        index_type=index_type,
        prefix="txt::"
    )


@timeit
def create_full_img_index(index_type):
    if index_type not in ["FLAT", "HNSW"]:
        raise IndexError("invalid index_type - must be FLAT or HNSW")
    idx_name = "idx_img" if index_type == "HNSW" else "idx_img_flat"
    print(f"creating index {idx_name}")
    create_index(
        index_name=idx_name,
        distance_metric=sc.IMG_DISTANCE_METRIC,
        vector_field_name="embedding",
        embedding_dimension=sc.IMG_EMBEDDING_DIMENSION,
        index_type=index_type,
        prefix="img::"
    )


if __name__ == '__main__':
    sc.api_redis_cli = sc.start_queueing(manually=True)
    sc.api_logger = sc.start_encoder_logging()
    # delete("txt:*")
    # delete("img:*")
    # to_redis(key_prefix="txt", file_name="txt_embeddings.parquet")
    # to_redis(key_prefix="img", file_name="img_embeddings.parquet")
    for bucket in range(5):
        create_bucket_txt_index(bucket)
        create_bucket_img_index(bucket)
    create_full_txt_index("FLAT")
    create_full_txt_index("HNSW")
    create_full_img_index("FLAT")
    create_full_img_index("HNSW")
