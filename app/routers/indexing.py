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


def publish_special_messages(token):
    for queue_id in range(sc.QUEUES_AVAILABLE):
        queue_txt_name = f"{sc.QUEUE_TXT}_{queue_id}"
        queue_img_name = f"{sc.QUEUE_IMG}_{queue_id}"
        sc.api_redis_cli.xadd(queue_txt_name, {"data": token.encode()})
        sc.api_redis_cli.xadd(queue_img_name, {"data": token.encode()})


@router.post("/")
def index(file: bytes = File(...), skip: int = 0):
    try:
        df = pd.read_csv(BytesIO(file))
        sc.api_logger.info("reading file and inserting into redis as streams")
        publish_special_messages(sc.START_TOKEN)
        for idx, row in df.iterrows():
            if int(str(idx)) < skip:  # in case of reprocessing
                continue
            queue_id = int(str(idx)) % sc.QUEUES_AVAILABLE
            queue_txt_name = f"{sc.QUEUE_TXT}_{queue_id}"
            queue_img_name = f"{sc.QUEUE_IMG}_{queue_id}"
            img_id = row['id']
            img_caption = row['caption']
            img_url = row['image']
            sc.api_logger.info(f"id: {img_id} | url: {img_url} | caption: {img_caption[:30]}...")
            sc.api_redis_cli.xadd(queue_txt_name, {"data": f"{img_id}{sc.SEPARATOR}{img_caption}".encode()})
            sc.api_redis_cli.xadd(queue_img_name, {"data": f"{img_id}{sc.SEPARATOR}{img_url}".encode()})
            sc.api_logger.info("inserted into redis")
        publish_special_messages(sc.END_TOKEN)
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
