# Vector search scaling (pgvector)

Reproduce: `make db && DATABASE_URL=postgresql://docspilot:docspilot@127.0.0.1:5433/docspilot \
uv run python -m docspilot.bench.vector_bench -n 50000 --noise 1.0` (and `--noise 3.0`).

**Setup**: 50,000 vectors x 1024 dims (the production schema), Gaussian-mixture data with 300
clusters, 100 queries that are perturbed copies of stored vectors (so each has a
clearly-nearest document, like real retrieval), k=10, recall measured against exact search.
Postgres 17 + pgvector in Docker on a laptop. Absolute numbers are *directional*: rerun on a
sample of your own corpus before deciding. Table size on disk: 281 MB.

## Strong cluster structure (`--noise 1.0`)

| config | build s | index MB | recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| exact (seq scan) | 0.0 | 0 | 1.000 | 162.1 | 167.1 |
| ivfflat lists=223 probes=1 | 5.7 | 411 | 1.000 | 1.0 | 1.5 |
| ivfflat lists=223 probes=10 | 5.7 | 411 | 1.000 | 3.1 | 3.9 |
| ivfflat lists=223 probes=30 | 5.7 | 411 | 1.000 | 7.9 | 9.2 |
| hnsw m=16 ef_construction=64 ef_search=10 | 31.1 | 410 | 0.821 | 1.3 | 1.8 |
| hnsw m=16 ef_construction=64 ef_search=40 | 31.1 | 410 | 0.970 | 1.5 | 2.2 |
| hnsw m=16 ef_construction=64 ef_search=100 | 31.1 | 410 | 1.000 | 1.9 | 2.6 |
| hnsw m=16 ef_construction=64 ef_search=200 | 31.1 | 410 | 1.000 | 3.6 | 4.9 |
| hnsw halfvec ef_search=40 | 17.8 | 137 | 0.990 | 0.9 | 1.5 |
| hnsw halfvec ef_search=100 | 17.8 | 137 | 1.000 | 0.9 | 1.2 |

## Weak cluster structure (`--noise 3.0`): the harder, more realistic case

| config | build s | index MB | recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| exact (seq scan) | 0.0 | 0 | 1.000 | 167.4 | 177.0 |
| ivfflat lists=223 probes=1 | 8.9 | 411 | 0.574 | 1.1 | 1.5 |
| ivfflat lists=223 probes=10 | 8.9 | 411 | 0.683 | 3.2 | 3.9 |
| ivfflat lists=223 probes=30 | 8.9 | 411 | 0.760 | 7.9 | 8.7 |
| hnsw m=16 ef_construction=64 ef_search=10 | 52.7 | 410 | 0.562 | 1.7 | 2.2 |
| hnsw m=16 ef_construction=64 ef_search=40 | 52.7 | 410 | 0.944 | 2.5 | 4.8 |
| hnsw m=16 ef_construction=64 ef_search=100 | 52.7 | 410 | 0.973 | 4.2 | 6.2 |
| hnsw m=16 ef_construction=64 ef_search=200 | 52.7 | 410 | 0.982 | 8.7 | 10.5 |
| hnsw halfvec ef_search=40 | 30.5 | 137 | 0.912 | 1.3 | 2.0 |
| hnsw halfvec ef_search=100 | 30.5 | 137 | 0.969 | 1.7 | 2.6 |

## What to take from it

- **Exact search is fine until it is not.** ~165 ms for 50k x 1024 is linear in N: 1M vectors
  would be ~3 s. The corpus in this repo (109 chunks) needs no index at all; indexes start
  to matter around 10^5 vectors.
- **IVFFlat** builds fast (6-9 s) but its recall depends on how well `lists` matches the data's
  structure: perfect on well-separated clusters, 0.57-0.76 on diffuse data even at 30 probes.
  It also needs representative data *at build time* and a rebuild as data drifts.
- **HNSW** costs 5-6x the build time (31-53 s) and the same ~410 MB, but reaches 0.94 recall at
  `ef_search=40` and 0.97-0.98 at 100-200 on the hard dataset, with 2.5-4 ms latency. It
  supports incremental inserts without retraining. Default choice for a retrieval workload.
- **`ef_search` is the runtime knob**: recall climbs and latency grows with it (10 -> 200
  moved recall 0.56 -> 0.98 and p50 1.7 -> 8.7 ms). Tune it against your *eval set* (Phase 4
  retrieval recall), not against a synthetic benchmark.
- **Half precision (`halfvec`) is nearly free**: the index is 3x smaller (137 vs 410 MB) and
  builds ~40% faster, for ~0.01-0.03 recall at the same `ef_search`. At 1M vectors that is
  ~2.7 GB vs ~8 GB of index to keep in RAM.
- **RAM is the real constraint**: an HNSW index that does not fit in memory degrades from
  milliseconds to disk-bound latencies. Budget index size x ~1.2 plus the working set.

## When to leave Postgres

Stay on pgvector while: one database to operate is worth more than the last 2x in vector
throughput; N is below ~10^7 vectors per index; filters are mostly relational (joins with
your tenant/ACL tables are the pgvector superpower); recall at your latency budget is met.

Consider a dedicated vector store (Qdrant, Weaviate, Pinecone, Vespa) when: you need
hundreds of millions of vectors or very high QPS with strict p99; you need native
multi-vector/sparse-dense hybrids at scale; vector load interferes with your OLTP database;
or you need online resharding. Measure the migration against the cost of simply
partitioning and adding a read replica first.

## Caveats of this benchmark

Synthetic data; one machine; cold/warm cache effects not separated; no concurrent writers;
single-query latency (not throughput under concurrency); no filtered search. Treat it as a
method and a starting point.
