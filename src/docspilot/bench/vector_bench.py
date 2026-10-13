"""pgvector scaling benchmark: exact vs IVFFlat vs HNSW vs HNSW on half-precision.

    DATABASE_URL=postgresql://... uv run python -m docspilot.bench.vector_bench -n 50000

What it answers: at *my* data size and recall target, which index, with which
parameters, costs how much build time / disk / latency, and when is a dedicated vector
database worth leaving Postgres for. It runs in a throwaway schema and cleans up.

Data caveat: purely random vectors are a worst case for ANN indexes (no structure to
exploit). Real embeddings cluster, so we generate a Gaussian mixture, which is much
closer, but absolute numbers still differ from your corpus: rerun on a sample of it."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.sql import SQL, Identifier


@dataclass
class Row:
    name: str
    build_s: float
    index_mb: float
    recall: float
    p50_ms: float
    p95_ms: float
    note: str = ""


def clustered(
    n: int, dim: int, clusters: int, rng: np.random.Generator, noise_scale: float = 1.0
) -> np.ndarray:
    centres = rng.normal(size=(clusters, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    assign = rng.integers(0, clusters, size=n)
    # per-dimension noise std 1.0/sqrt(dim) -> noise norm ~1: members of a cluster are
    # clearly related but individually distinguishable (cos ~0.5 to the centre)
    noise = rng.normal(scale=noise_scale / math.sqrt(dim), size=(n, dim)).astype(np.float32)
    x = centres[assign] + noise
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def scalar(cursor: Any) -> Any:
    row = cursor.fetchone()
    assert row is not None
    return row[0]


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dsn = args.dsn or os.environ["DATABASE_URL"]
    schema = f"bench_{secrets.token_hex(4)}"
    rng = np.random.default_rng(args.seed)
    data = clustered(args.n, args.dim, args.clusters, rng, args.noise)
    # Queries are perturbed copies of stored vectors: like a real search, each query has a
    # clearly-nearest document. Fully independent queries would sit equidistant from a whole
    # cluster, so "recall" would mostly measure tie-breaking noise.
    picks = rng.integers(0, args.n, size=args.queries)
    queries = data[picks] + rng.normal(
        scale=0.4 / math.sqrt(args.dim), size=(args.queries, args.dim)
    ).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    k = args.k

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
        conn.execute(SQL("SET search_path = {}, public").format(Identifier(schema)))
        conn.execute(f"CREATE TABLE v (id int PRIMARY KEY, embedding vector({args.dim}))")
        t0 = time.perf_counter()
        with (
            conn.cursor() as cur,
            cur.copy("COPY v (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy,
        ):
            copy.set_types(["int4", "vector"])
            for i, vec in enumerate(data):
                copy.write_row((i, vec))
        load_s = time.perf_counter() - t0
        conn.execute("ANALYZE v")
        conn.execute("SET maintenance_work_mem = '1GB'")
        conn.execute("SET max_parallel_maintenance_workers = 4")
        table_mb = scalar(conn.execute("SELECT pg_total_relation_size('v')/1e6"))
        print(f"loaded {args.n} x {args.dim} in {load_s:.1f}s, table {table_mb:.0f} MB")

        def search_sql(expr: str, cast: str = "") -> str:
            return f"SELECT id FROM v ORDER BY {expr} <=> %s{cast} LIMIT {k}"

        def truth_for(q: np.ndarray) -> set[int]:
            return {r[0] for r in conn.execute(search_sql("embedding"), (q,)).fetchall()}

        # exact ground truth (no index yet => sequential scan) and its latency
        truths: list[set[int]] = []
        exact_ms: list[float] = []
        for q in queries:
            t = time.perf_counter()
            truths.append(truth_for(q))
            exact_ms.append((time.perf_counter() - t) * 1000)
        rows = [
            Row(
                "exact (seq scan)",
                0.0,
                0.0,
                1.0,
                statistics.median(exact_ms),
                percentile(exact_ms, 0.95),
            )
        ]

        def bench(
            label: str,
            sql: str,
            cast: Any,
            build_s: float,
            index_mb: float,
            note: str,
            settings: list[str],
        ) -> Row:
            for s in settings:
                conn.execute(s)
            lat: list[float] = []
            hits = 0
            for q, truth in zip(queries, truths, strict=True):
                qq = cast(q)
                t = time.perf_counter()
                got = {r[0] for r in conn.execute(sql, (qq,)).fetchall()}
                lat.append((time.perf_counter() - t) * 1000)
                hits += len(got & truth)
            return Row(
                label,
                build_s,
                index_mb,
                hits / (k * len(queries)),
                statistics.median(lat),
                percentile(lat, 0.95),
                note,
            )

        def index_size(name: str) -> float:
            return float(scalar(conn.execute("SELECT pg_relation_size(%s::regclass)/1e6", (name,))))

        ident = lambda q: q
        # --- IVFFlat -------------------------------------------------------------
        lists = max(10, int(math.sqrt(args.n)))
        t = time.perf_counter()
        conn.execute(
            f"CREATE INDEX ivf ON v USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        )
        build = time.perf_counter() - t
        size = index_size("ivf")
        for probes in (1, 10, 30):
            rows.append(
                bench(
                    f"ivfflat lists={lists} probes={probes}",
                    search_sql("embedding"),
                    ident,
                    build,
                    size,
                    "needs training data; rebuild as data drifts",
                    [f"SET ivfflat.probes = {probes}"],
                )
            )
        conn.execute("DROP INDEX ivf")

        # --- HNSW ----------------------------------------------------------------
        t = time.perf_counter()
        conn.execute(
            "CREATE INDEX hn ON v USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        build = time.perf_counter() - t
        size = index_size("hn")
        for ef in (10, 40, 100, 200):
            rows.append(
                bench(
                    f"hnsw m=16 ef_construction=64 ef_search={ef}",
                    search_sql("embedding"),
                    ident,
                    build,
                    size,
                    "incremental inserts; larger RAM",
                    [f"SET hnsw.ef_search = {ef}"],
                )
            )
        plan = conn.execute("EXPLAIN " + search_sql("embedding"), (queries[0],)).fetchall()
        used_index = any("hn" in str(r) for r in plan)
        conn.execute("DROP INDEX hn")

        # --- HNSW on half precision (scalar quantisation) -------------------------
        t = time.perf_counter()
        conn.execute(
            f"CREATE INDEX hn_half ON v USING hnsw ((embedding::halfvec({args.dim})) halfvec_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        build = time.perf_counter() - t
        size = index_size("hn_half")
        half_sql = f"SELECT id FROM v ORDER BY embedding::halfvec({args.dim}) <=> %s::halfvec({args.dim}) LIMIT {k}"
        for ef in (40, 100):
            rows.append(
                bench(
                    f"hnsw halfvec ef_search={ef}",
                    half_sql,
                    ident,
                    build,
                    size,
                    "half the index size; tiny recall cost",
                    [f"SET hnsw.ef_search = {ef}"],
                )
            )
        conn.execute("DROP INDEX hn_half")

        conn.execute(SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema)))

    return {
        "n": args.n,
        "noise": args.noise,
        "dim": args.dim,
        "queries": args.queries,
        "k": k,
        "load_s": round(load_s, 1),
        "table_mb": round(float(table_mb), 1),
        "hnsw_used_index": used_index,
        "rows": [asdict(r) for r in rows],
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        f"n={result['n']:,} dim={result['dim']} noise={result['noise']} k={result['k']} queries={result['queries']} "
        f"table={result['table_mb']} MB",
        "",
        "| config | build s | index MB | recall@k | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['name']} | {r['build_s']:.1f} | {r['index_mb']:.0f} | {r['recall']:.3f} "
            f"| {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--clusters", type=int, default=300)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--noise", type=float, default=1.0, help="cluster spread; higher = weaker structure"
    )
    ap.add_argument("--dsn")
    ap.add_argument("--out")
    args = ap.parse_args()
    result = run(args)
    print(render(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
