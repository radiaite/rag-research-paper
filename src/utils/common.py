"""Shared utilities: logging, timing, config loading, reproducibility."""

import contextlib
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = DATA_DIR / "results"


def set_seed(seed: int = 42) -> None:
    """Set random seeds + deterministic CUDA flags for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = CONFIGS_DIR / "default.yaml"
    return yaml.safe_load(Path(config_path).read_text())


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    return logger


@dataclass
class RetrievedDoc:
    """A single retrieved document with metadata."""

    doc_id: str
    text: str
    score: float
    rank: int
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Stores results for a single experiment run."""

    method: str
    config: dict[str, Any]
    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, float] | None = None
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    index_time_seconds: float = 0.0
    index_size_mb: float = 0.0
    num_queries: int = 0
    avg_latency_ms: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(**json.loads(path.read_text()))


def _safe_version(module_name: str) -> str | None:
    """Return importlib.metadata version for a package, or None if not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(module_name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _file_sha256(path: Path) -> str | None:
    """Compute SHA-256 of a file. Returns None if the file does not exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_provenance(
    embedding_model: str | None = None,
    index_path: Path | str | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata for an experiment run."""
    prov: dict[str, Any] = {}

    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        prov["git_sha"] = sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        prov["git_sha"] = None

    prov["torch_version"] = _safe_version("torch")
    prov["transformers_version"] = _safe_version("transformers")
    prov["sentence_transformers_version"] = _safe_version("sentence-transformers")
    prov["faiss_version"] = _safe_version("faiss-cpu") or _safe_version("faiss-gpu")

    try:
        import torch

        prov["cuda_available"] = bool(torch.cuda.is_available())
        prov["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        prov["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        prov["cuda_available"] = False
        prov["cuda_version"] = None
        prov["gpu_name"] = None

    # Best-effort: skipped if no network.
    if embedding_model:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(embedding_model)
            prov["embedding_model_revision"] = getattr(info, "sha", None)
        except Exception:
            prov["embedding_model_revision"] = None

    if index_path is not None:
        ip = Path(index_path)
        if ip.is_dir():
            ip = ip / "index.faiss"
        prov["index_faiss_sha256"] = _file_sha256(ip)

    return prov


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000
