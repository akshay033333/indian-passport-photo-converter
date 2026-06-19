"""ASGI entry point: Streamlit app plus SEO routes for crawlers and Search Console."""

from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse, PlainTextResponse, Response
from starlette.routing import Route
from streamlit.starlette import App

CANONICAL_URL = "https://passportphoto-converter.streamlit.app"
SEO_DIR = Path(__file__).parent / "seo"


async def robots_txt(_request) -> PlainTextResponse:
    body = f"User-agent: *\nAllow: /\nSitemap: {CANONICAL_URL}/sitemap.xml\n"
    return PlainTextResponse(body)


async def sitemap_xml(_request) -> Response:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{CANONICAL_URL}/</loc></url>
</urlset>"""
    return Response(xml, media_type="application/xml")


def _gsc_verification_handler(path: Path):
    async def handler(_request) -> FileResponse:
        return FileResponse(path, media_type="text/html")

    return handler


def _gsc_routes() -> list[Route]:
    routes: list[Route] = []
    if not SEO_DIR.is_dir():
        return routes
    for path in sorted(SEO_DIR.glob("google*.html")):
        routes.append(Route(f"/{path.name}", _gsc_verification_handler(path)))
    return routes


app = App(
    "app.py",
    routes=[
        Route("/robots.txt", robots_txt),
        Route("/sitemap.xml", sitemap_xml),
        *_gsc_routes(),
    ],
)
