# gigacity-central

C owns admission, VLM serving, and budgeting within the C/D team. Serving engine, media ingestion, and feedback semantics remain open. Import shared types from `gigacity_contracts`. Add unit tests under `tests/`.

## Run the draft endpoint locally

From the repository root, install the workspace and start the synthetic backend.

```bash
uv sync --locked --all-packages
uv run --no-sync python -m gigacity_central.server --synthetic --port 8200
```

From another terminal at the repository root:

```bash
curl -fsS http://127.0.0.1:8200/health
curl -fsS http://127.0.0.1:8200/v1/infer \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/inference_request.json
```

The response contains the request's IDs and a synthetic Yes answer. See
[the draft contract](../../docs/contracts.md#draft-http-endpoint).
