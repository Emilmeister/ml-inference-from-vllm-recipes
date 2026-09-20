from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vllm_cloudru.api import router
from vllm_cloudru.cloudru import CloudruClient
from vllm_cloudru.models import CloudCredentials
from vllm_cloudru.recipes import RecipeClient

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(
    recipes: Any | None = None,
    cloud_factory: Callable[[CloudCredentials], Any] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI) -> AsyncIterator[None]:
        recipe_client = recipes if recipes is not None else RecipeClient()
        cloud_http: httpx.AsyncClient | None = None
        resolved_cloud_factory = cloud_factory
        if resolved_cloud_factory is None:
            cloud_http = httpx.AsyncClient(timeout=20)

            def resolved_cloud_factory(credentials: CloudCredentials) -> CloudruClient:
                assert cloud_http is not None
                return CloudruClient(credentials, cloud_http)

        lifespan_app.state.recipes = recipe_client
        lifespan_app.state.cloud_factory = resolved_cloud_factory
        try:
            yield
        finally:
            try:
                if recipes is None:
                    await recipe_client.aclose()
            finally:
                if cloud_http is not None:
                    await cloud_http.aclose()

    app = FastAPI(title="vLLM Recipes → Cloud.ru", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Any:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        # AICODE-NOTE: FastAPI's default 422 body echoes rejected input, which can
        # contain Cloud.ru credentials. Return field locations only.
        fields = sorted(
            {
                ".".join(str(part) for part in issue["loc"] if part != "body")
                for issue in error.errors()
            }
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "Проверьте поля запроса",
                    "fields": fields,
                }
            },
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    assets = FRONTEND_DIST / "assets"
    illustrations = FRONTEND_DIST / "illustrations"
    index = FRONTEND_DIST / "index.html"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")
    if illustrations.is_dir():
        app.mount(
            "/illustrations",
            StaticFiles(directory=illustrations),
            name="frontend-illustrations",
        )

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        known_spa_path = path.startswith("catalog/")
        if (
            path.startswith("api/")
            or (Path(path).suffix and not known_spa_path)
            or not index.is_file()
        ):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(index)

    return app


app = create_app()
