from fastapi import APIRouter, File, HTTPException
import traceback
import pandas as pd
import app.shared_context as sc
from io import BytesIO


router = APIRouter(
    prefix="/index",
    tags=["indexing"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
def index(file: bytes = File(...), skip: int = 0):
    try:
        df = pd.read_csv(BytesIO(file))
        sc.api_logger.info("reading file and inserting into redis as pubsub")
        for idx, row in df.iterrows():
            if int(str(idx)) < skip:  # in case of reprocessing
                continue
            img_id = row['id']
            img_caption = row['caption']
            img_url = row['image']
            sc.api_logger.info(f"id: {img_id} | url: {img_url} | caption: {img_caption}")
            sc.api_logger.info("starting redis insertion")
            sc.api_redis_cli.publish(sc.QUEUE_TXT, f"{img_id}|{img_caption}".encode())
            sc.api_redis_cli.publish(sc.QUEUE_IMG, f"{img_id}|{img_url}".encode())
            sc.api_logger.info("ending redis insertion")
        return {"msg": f"{df.shape[0]} files inserted"}
    except (TypeError, OSError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Index Route: Error reading CSV file - {str(exc)}"
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
