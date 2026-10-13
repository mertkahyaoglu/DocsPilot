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
| exact (seq scan) | 0.0 | 0 | 1.000 | 163.0 | 169.2 |
| ivfflat lists=223 probes=1 | 5.3 | 411 | 0.998 | 1.0 | 1.4 |
| ivfflat lists=223 probes=10 | 5.3 | 411 | 1.000 | 3.1 | 3.9 |
| ivfflat lists=223 probes=30 | 5.3 | 411 | 1.000 | 7.9 | 8.9 |
| hnsw m=16 ef_construction=64 ef_search=10 | 31.1 | 410 | 0.823 | 1.3 | 1.8 |
| hnsw m=16 ef_construction=64 ef_search=40 | 31.1 | 410 | 0.990 | 1.5 | 2.4 |
| hnsw m=16 ef_construction=64 ef_search=100 | 31.1 | 410 | 1.000 | 1.9 | 2.7 |
| hnsw m=16 ef_construction=64 ef_search=200 | 31.1 | 410 | 1.000 | 3.7 | 4.6 |
| hnsw halfvec ef_search=40 | 17.9 | 137 | 0.990 | 0.9 | 1.2 |
| hnsw halfvec ef_search=100 | 17.9 | 137 | 1.000 | 1.0 | 1.3 |

## Weak cluster structure (`--noise 3.0`): the harder, more realistic case

| config | build s | index MB | recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| exact (seq scan) | 0.0 | 0 | 1.000 | 162.9 | 169.8 |
| ivfflat lists=223 probes=1 | 8.3 | 411 | 0.692 | 1.0 | 1.2 |
| ivfflat lists=223 probes=10 | 8.3 | 411 | 0.762 | 3.1 | 3.6 |
| ivfflat lists=223 probes=30 | 8.3 | 411 | 0.818 | 7.9 | 8.5 |
| hnsw m=16 ef_construction=64 ef_search=10 | 52.7 | 410 | 0.688 | 1.5 | 2.2 |
| hnsw m=16 ef_construction=64 ef_search=40 | 52.7 | 410 | 0.937 | 2.4 | 4.1 |
| hnsw m=16 ef_construction=64 ef_search=100 | 52.7 | 410 | 0.977 | 3.9 | 5.9 |
| hnsw m=16 ef_construction=64 ef_search=200 | 52.7 | 410 | 0.980 | 8.5 | 9.9 |
| hnsw halfvec ef_search=40 | 30.9 | 137 | 0.926 | 1.3 | 2.3 |
| hnsw halfvec ef_search=100 | 30.9 | 137 | 0.975 | 1.7 | 2.5 |

## What to take from it

- **Exact search is fine until it is not.** ~165 ms for 50k x 1024 is linear in N: 1M vectors
  would be ~3 s. The corpus in this repo (109 chunks) needs no index at all; indexes start
  to matter around 10^5 vectors.
- **IVFFlat** builds fast (5-9 s) but its recall depends on how well `lists` matches the data's
  structure: ~1.0 on well-separated clusters, only 0.69-0.82 on diffuse data even at 30
  probes. It also needs representative data *at build time* (k-means) and a rebuild as data
  drifts. Its recall varied by up to ~0.1 between two runs on the same data (random k-means
  initialisation): always run a benchmark more than once.
- **HNSW** costs 4-6x the build time (31-53 s) and the same ~410 MB, but reaches 0.94 recall at
  `ef_search=40` and 0.98 at 100-200 on the hard dataset, with 2.4-8.5 ms latency. It
  supports incremental inserts without retraining. Default choice for a retrieval workload.
- **`ef_search` is the runtime knob**: recall climbs and latency grows with it (10 -> 200
  moved recall 0.69 -> 0.98 and p50 1.5 -> 8.5 ms). Tune it against your *eval set* (Phase 4
  retrieval recall), not against a synthetic benchmark.
- **Half precision (`halfvec`) is nearly free**: the index is 3x smaller (137 vs 410 MB; more
  than the 2x the bytes alone suggest because of how 4 KB vectors pack into 8 KB pages) and
  builds ~40% faster, for ~0.01 recall at the same `ef_search`. At 1M vectors that is
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

Every index is `EXPLAIN`-checked to be the one actually used (the run flags it if not).
The benchmark schema is dropped even if a build fails.

Synthetic data; one machine; cold/warm cache effects not separated; no concurrent writers;
single-query latency (not throughput under concurrency); no filtered search. Treat it as a
method and a starting point.
