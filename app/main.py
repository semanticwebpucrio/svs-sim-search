import redis
import uvicorn
import app.shared_context as sc
from fastapi import FastAPI
from app.routers import indexing


def get_application() -> FastAPI:
    app = FastAPI(
        title="Multimodal Clustering",
        description=sc.API_DESCRIPTION,
        version="0.0.1",
    )
    app.include_router(indexing.router)

    # TODO: receive a CSV file to be parsed and inserted into redis queue
    #  consumers will be responsible to read each message, apply model, and persist the result into redis again
    @app.on_event("startup")
    def startup_event():
        sc.redis_cli = redis.Redis(
            host=sc.REDIS_HOST,
            port=sc.REDIS_PORT,
        )
        sc.sub_pre_text = sc.redis_cli.pubsub()
        sc.sub_pre_text.subscribe("pre_text_queue")
        sc.sub_text = sc.redis_cli.pubsub()
        sc.sub_text.subscribe("text_queue")
        sc.sub_pre_image = sc.redis_cli.pubsub()
        sc.sub_pre_image.subscribe("pre_image_queue")
        sc.sub_image = sc.redis_cli.pubsub()
        sc.sub_image.subscribe("image_queue")

    @app.get("/healthcheck", include_in_schema=False)
    def healthcheck():
        return {"status": "ok"}

    return app


sc.api_app = get_application()


if __name__ == '__main__':
    uvicorn.run(sc.api_app, host=sc.API_HOST, port=sc.API_PORT)
