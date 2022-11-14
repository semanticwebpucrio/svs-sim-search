import io
import redis
import traceback
from PIL import Image
import app.shared_context as sc
from fastapi import APIRouter, File, HTTPException
from redis.commands.search.query import Query


router = APIRouter(
    prefix="/search",
    tags=["searching"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
def index(kws: str = None, img: bytes = File(default=None), k: int = 5):
    try:
        result_txt = retrieve_txt(kws, k) if kws else {}
        result_img = retrieve_img(img, k) if img else {}
        result = {
            "total": result_txt.get("total", 0) + result_img.get("total", 0),
            "results": result_txt.get("results", []) + result_img.get("results", [])
        }
        return result
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


def retrieve_txt(kws: str, k: int):
    query_vector = sc.load_txt_model().encode(kws).astype(sc.TEXT_EMBEDDING_TYPE).tobytes()
    # TODO: pay attention on asc and desc which may vary depending on distance metric
    q = Query(f"*=>[KNN {k} @embedding $query_vector AS score]")\
        .sort_by("score", asc=False)\
        .return_fields("id", "sentence", "score")\
        .dialect(2)
    params_dict = {"query_vector": query_vector}
    results = sc.api_redis_cli.ft(index_name=sc.TEXT_INDEX_NAME).search(q, query_params=params_dict)
    ret = {
        "total": results.total,
        "results": [
            {"id": doc.id, "sentence": doc.sentence, "score": doc.score} for doc in results.docs
        ]
    }
    return ret


def retrieve_img(raw: bytes, k: int):
    image = Image.open(io.BytesIO(raw))
    embeddings = sc.encode_image(input_image=image)
    query_vector = embeddings.detach().numpy().astype(sc.IMG_EMBEDDING_TYPE).tobytes()
    # TODO: pay attention on asc and desc which may vary depending on distance metric
    q = Query(f"*=>[KNN {k} @{sc.IMG_EMBEDDING_FIELD_NAME} $query_vector AS score]") \
        .sort_by("score", asc=False) \
        .return_fields("id", "sentence", "score") \
        .dialect(2)
    # FIXME: Error parsing vector similarity query
    #  query vector blob size (4000) does not match index's expected size (3072).
    params_dict = {"query_vector": query_vector[:3072]}
    results = sc.api_redis_cli.ft(index_name=sc.IMG_INDEX_NAME).search(q, query_params=params_dict)
    ret = {
        "total": results.total,
        "results": [
            {"id": doc.id, "score": doc.score} for doc in results.docs
        ]
    }
    return ret
