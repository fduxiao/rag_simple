#!/bin/sh
uv tool run ruff format && \
uv tool run ruff check --select=FIX
