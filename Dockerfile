# pcrouter: the routing proxy (and the fake worker / replay client for demos).
#   docker build -t pcrouter .
#   docker run --rm -p 8080:8080 pcrouter serve --worker http://gpu-0:8000 --worker http://gpu-1:8000
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server]" uvloop && useradd --system --no-create-home pcrouter
USER pcrouter
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)" || exit 1
ENTRYPOINT ["pcrouter"]
CMD ["serve", "--help"]
