#!/usr/bin/env bash
# BGE-M3 + ColBERTv2 runs on a rented GPU; skips runs whose result JSON is complete.
# Flags: --sanity-only --force --skip-install --skip-sanity. Output tee'd to run.log.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

LOG_FILE="$REPO_ROOT/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

SANITY_ONLY=0
FORCE=0
SKIP_INSTALL=0
SKIP_SANITY=0

for arg in "$@"; do
    case "$arg" in
        --sanity-only)   SANITY_ONLY=1 ;;
        --force)         FORCE=1 ;;
        --skip-install)  SKIP_INSTALL=1 ;;
        --skip-sanity)   SKIP_SANITY=1 ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg  (try --help)"; exit 1 ;;
    esac
done

log()   { echo "[$(date +%H:%M:%S)] $*"; }
phase() {
    echo
    echo "================================================================="
    echo " $*"
    echo "================================================================="
}

trap 'echo; echo "Run log saved to: $LOG_FILE"' EXIT

phase "Phase A — environment check"

python3 - <<'PY'
import sys, platform
print(f"Python      : {sys.version.split()[0]}")
print(f"Platform    : {platform.platform()}")
try:
    import torch
    print(f"Torch       : {torch.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA        : {torch.version.cuda}")
        print(f"GPU         : {gpu}  (sm_{cap[0]}{cap[1]})")
        print(f"GPU memory  : {gb:.1f} GB")
        # Batch size follows memory: reranker peaks ~15 GB at batch 128.
        if gb >= 40:
            print("GPU check   : OK (≥40 GB; default batch sizes safe)")
        elif gb >= 24:
            print(f"WARNING     : {gb:.1f} GB GPU. Default rerankers.bge.batch_size=128 "
                  "may OOM under load; consider 32-64 in configs/default.yaml.")
        else:
            print(f"WARNING     : {gb:.1f} GB GPU is small. Lower "
                  "rerankers.bge.batch_size to 8 in configs/default.yaml.")
    else:
        print("!! WARNING  : no CUDA. The reranker pass would take days on CPU.")
except ImportError:
    print("Torch not installed yet (will be installed in Phase B).")
PY

if [ "$SKIP_INSTALL" -eq 0 ]; then
    phase "Phase B — install pinned dependencies"
    if [ ! -f requirements-h100.txt ]; then
        echo "ERROR: requirements-h100.txt not found in $REPO_ROOT"
        exit 2
    fi
    pip install -q -r requirements-h100.txt
    pip install -q -e .
    log "deps installed"
else
    phase "Phase B — SKIPPED (--skip-install)"
fi

if [ "$SKIP_INSTALL" -eq 0 ] && ! python3 -c "import ragatouille" 2>/dev/null; then
    if [ -f "$REPO_ROOT/data/results/colbertv2_whole_doc.json" ]; then
        log "Phase B' — ColBERTv2 result already present; skipping ragatouille install"
    else
        phase "Phase B' — install ragatouille (--no-deps to protect existing torch)"
        # setup.py pins torch ranges that clash with cu128; <0.0.10 keeps the Stanford ColBERT backend.
        pip install -q --no-deps "ragatouille>=0.0.9,<0.0.10"
        pip install -q --no-deps "colbert-ai>=0.2.21"
        # ragatouille imports langchain.retrievers at load time.
        pip install -q "bitarray>=2.9" "git-python>=1.0" "ujson>=5.0" \
                       "ninja>=1.11" "voyager>=2.0" "fast-pytorch-kmeans>=0.2" \
                       "langchain>=0.1,<0.4" "langchain-community>=0.0.20" \
                       "llama-index-core>=0.10" "accelerate>=0.30"
        python3 -c "from ragatouille import RAGPretrainedModel" \
            || { log "ragatouille import failed; ColBERTv2 phase will fail noisily"; }
        log "ragatouille installed (no torch downgrade)"
    fi
fi

phase "Phase C — download T²-RAGBench (cached after first call)"
python3 - <<'PY'
from src.data_loader import load_t2ragbench
data = load_t2ragbench()
assert data.num_queries   == 23088, f"Expected 23088 queries, got {data.num_queries}"
assert data.num_documents ==  7318, f"Expected 7318 documents, got {data.num_documents}"
print(data.summary())
PY

if [ "$SKIP_SANITY" -eq 0 ]; then
    phase "Phase D — sanity (100 queries, hybrid + bge_m3 + bge rerank)"
    python3 scripts/run_experiment.py \
        --method hybrid --embedding bge_m3 --reranker bge \
        --top-k 20 --max-queries 100 \
        --output-name sanity_hybrid+bge_bge_m3

    python3 - <<'PY'
import json
d = json.load(open("data/results/sanity_hybrid+bge_bge_m3.json"))
n = len(d["per_query_results"])
m = d["retrieval_metrics"]
print(f"sanity records: {n}")
for k in ("recall@1","recall@5","recall@10","mrr@3","ndcg@10"):
    if k in m: print(f"  {k}: {m[k]:.4f}")
assert n == 100, "sanity did not produce 100 per-query records"
PY
    log "sanity ok"
else
    phase "Phase D — SKIPPED (--skip-sanity)"
fi

if [ "$SANITY_ONLY" -eq 1 ]; then
    phase "Sanity-only mode — exiting before full runs."
    exit 0
fi

phase "Phase E — detect missing runs"

# Returns 0 (run is needed) if file missing, incomplete, or --force.
needs_run() {
    local result_file="data/results/$1"
    if [ "$FORCE" -eq 1 ]; then return 0; fi
    if [ ! -f "$result_file" ]; then return 0; fi
    if python3 - "$result_file" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    sys.exit(0 if len(d.get("per_query_results", [])) == 23088 else 1)
except Exception:
    sys.exit(1)
PY
    then
        return 1   # file is complete -> skip
    else
        return 0   # incomplete -> run
    fi
}

run_or_skip() {
    local label="$1"; shift
    local outfile="$1"; shift
    if needs_run "$outfile"; then
        log "RUN  : $label   ->   data/results/$outfile"
        python3 scripts/run_experiment.py "$@"
    else
        log "SKIP : $label   (already at data/results/$outfile, 23088 records)"
    fi
}

phase "Phase F — full retrieval runs"

run_or_skip "BGE-M3 dense" \
            "dense_bge_m3_whole_doc.json" \
            --method dense  --embedding bge_m3 --reranker none --top-k 20

run_or_skip "BGE-M3 hybrid (BM25 + BGE-M3, RRF)" \
            "hybrid_bge_m3_whole_doc.json" \
            --method hybrid --embedding bge_m3 --reranker none --top-k 20

run_or_skip "BGE-M3 hybrid + BGE rerank (HEADLINE)" \
            "hybrid+bge_bge_m3_whole_doc.json" \
            --method hybrid --embedding bge_m3 --reranker bge --top-k 20

# Override max_length/batch_size on the CLI: YAML-keyed rerankers silently fell back to NoReranker on older HEADs.
run_or_skip "BGE-M3 hybrid + BGE rerank @ max_length=1024 (in-regime sanity)" \
            "hybrid+bge_1024_bge_m3_whole_doc.json" \
            --method hybrid --embedding bge_m3 \
            --reranker bge --reranker-max-length 1024 --reranker-batch-size 64 \
            --output-name hybrid+bge_1024_bge_m3_whole_doc \
            --top-k 20

run_or_skip "BGE-M3 hybrid + BGE rerank @ max_length=2048 (out-of-regime sanity)" \
            "hybrid+bge_2048_bge_m3_whole_doc.json" \
            --method hybrid --embedding bge_m3 \
            --reranker bge --reranker-max-length 2048 --reranker-batch-size 32 \
            --output-name hybrid+bge_2048_bge_m3_whole_doc \
            --top-k 20

phase "Phase F' — ColBERTv2 (late-interaction)"

if needs_run "colbertv2_whole_doc.json"; then
    log "RUN  : ColBERTv2   ->   data/results/colbertv2_whole_doc.json"
    python3 scripts/run_colbertv2.py --top-k 20
else
    log "SKIP : ColBERTv2  (already at data/results/colbertv2_whole_doc.json, 23088 records)"
fi

phase "Phase G — provenance + summary"

pip freeze > provenance.log
log "wrote provenance.log ($(wc -l < provenance.log) packages)"

python3 - <<'PY'
import glob, json, os

print("=== BGE-M3 results summary ===")
files = sorted(set(
    glob.glob("data/results/*bge_m3*whole_doc.json") +
    glob.glob("data/results/hybrid+bge_bge_m3_whole_doc.json") +
    glob.glob("data/results/hybrid+bge_1024_bge_m3_whole_doc.json") +
    glob.glob("data/results/hybrid+bge_2048_bge_m3_whole_doc.json") +
    glob.glob("data/results/colbertv2_whole_doc.json")
))
for f in files:
    d = json.load(open(f))
    m = d["retrieval_metrics"]
    cfg = d["config"]
    p   = cfg.get("provenance", {}) or {}
    print()
    print(f"{f}")
    print(f"  config:   method={cfg['method']}  embedding={cfg['embedding']}  "
          f"reranker={cfg['reranker']}")
    print(f"  R@5 ={m.get('recall@5', 0):.4f}    R@10 ={m.get('recall@10', 0):.4f}    "
          f"MRR@3={m.get('mrr@3', 0):.4f}    nDCG@10={m.get('ndcg@10', 0):.4f}")
    print(f"  records: {len(d.get('per_query_results', []))}    "
          f"avg latency: {d.get('avg_latency_ms', 0):.1f} ms")
    print(f"  GPU: {p.get('gpu_name', '?')}    "
          f"git={(p.get('git_sha') or '?')[:8]}    "
          f"index_sha256={(p.get('index_faiss_sha256') or '?')[:16]}...")
PY

phase "DONE."
cat <<'EOF'

Next steps (run on your laptop, NOT on the rented GPU):

1. Sync the artifacts back. Pick one:

   # Option A — git push (preserves provenance in history)
   git checkout -b bge-m3-h100-runs
   git add data/results/*bge_m3*.json provenance.log
   git commit -m "Add BGE-M3 results from H100 run"
   git push origin bge-m3-h100-runs

   # Option B — rsync, from your laptop
   rsync -av <h100-host>:~/RAGPaper/data/results/*bge_m3*.json ./data/results/
   rsync -av <h100-host>:~/RAGPaper/provenance.log ./

2. Aggregate + significance + figures:
   python3 scripts/aggregate_bge_m3.py --out results/bge_m3_results.md
   python3 scripts/bge_m3_stats.py     --out results/bge_m3_results.md
   python3 scripts/generate_figures.py

3. Apply the LaTeX edits described in notebooks/AFTER_H100.md.

EOF
