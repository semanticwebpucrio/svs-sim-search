import uvicorn
import app.shared_context as sc
from fastapi import FastAPI
from app.routers import indexing, search


def get_application() -> FastAPI:
    app = FastAPI(
        title="Multimodal Clustering",
        description=sc.API_DESCRIPTION,
        version="0.0.1",
    )
    app.include_router(indexing.router)
    app.include_router(search.router)

    @app.on_event("startup")
    def startup_event():
        sc.api_redis_cli = sc.start_queueing()
        sc.api_logger = sc.start_api_logging()

    @app.get("/healthcheck", include_in_schema=False)
    def healthcheck():
        return {"status": "ok"}

    return app


sc.api_app = get_application()


if __name__ == '__main__':
    uvicorn.run(sc.api_app, host=sc.API_HOST, port=sc.API_PORT)
