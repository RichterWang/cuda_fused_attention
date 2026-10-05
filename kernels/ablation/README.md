# Controlled ablation kernels

Each future ablation must change one primary mechanism while preserving the
attention contract, shape, tile configuration where applicable, input
generation, correctness threshold, and timing protocol.

Planned variants:

- `v2_one_query_row.cu`: isolate cross-query K/V reuse.
- `v2_scalar_loads.cu`: isolate vectorized loads and stores.
- `v2_no_swizzle.cu`: isolate shared-memory layout/bank-conflict mitigation.
- `v2_sync_single_stage.cu`: establish the synchronous tiled pipeline baseline.

Do not add an ablation to the benchmark grid until it passes the same
correctness cases as V0/V1/V2.
