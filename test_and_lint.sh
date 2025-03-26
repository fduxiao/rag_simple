#/bin/sh

uv run -m unittest tests && \
uv tool run ruff check ./src && \
