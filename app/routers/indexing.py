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
        for idx, row in df.iterrows():
            if int(str(idx)) < skip:  # in case of reprocessing
                continue
            # list_id = row['id']
            img_caption = row['caption']
            img_url = row['image']
            sc.redis_cli.publish("pre_text_queue", img_caption)
            sc.redis_cli.publish("pre_img_queue", img_url)
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
