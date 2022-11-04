import redis
import traceback
from pathlib import Path
import app.shared_context as sc
from fastapi import APIRouter, HTTPException
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer


router = APIRouter(
    prefix="/search",
    tags=["searching"],
    responses={404: {"description": "Not found"}},
)


model = SentenceTransformer(
    model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    cache_folder=str(Path(__file__).parent.parent / "dl_models"),
)


@router.get("/")
def index(kws: str, k: int = 5):
    try:
        query_vector = model.encode(kws).astype(sc.TEXT_EMBEDDING_TYPE).tobytes()
        q = Query(f"*=>[KNN {k} @{sc.TEXT_EMBEDDING_FIELD_NAME} $query_vector AS score]")\
            .sort_by("score", asc=False)\
            .return_fields("id", "sentence", "score")\
            .dialect(2)
        params_dict = {"query_vector": query_vector}
        results = sc.api_redis_cli.ft(index_name="idx_txt").search(q, query_params=params_dict)
        ret = {
            "total": results.total,
            "results": [
                {"id": doc.id, "sentence": doc.sentence, "score": doc.score} for doc in results.docs
            ]
        }
        return ret
    except (NameError, redis.exceptions.ResponseError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Search Route: Error querying Redis - {str(exc)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="".join(
                traceback.format_exception(
                    etype=type(exc), value=exc, tb=exc.__traceback__
                )
            )
        )
