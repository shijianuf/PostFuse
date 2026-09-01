"""Dataset registry: CASIA (main) + EmoDB (second-corpus check)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from erm.datasets.casia import (
    EMOTION_LABELS as CASIA_EMOTION_LABELS,
    discover_casia_samples,
    split_samples as casia_split_samples,
)
from erm.datasets.emodb import (
    EMOTION_LABELS as EMODB_EMOTION_LABELS,
    discover_emodb_samples,
    split_samples as emodb_split_samples,
)

# english / emodb / emo_db / berlin are aliases for the same EmoDB loader
DATASET_CHOICES = ("casia", "emodb", "emo_db", "berlin", "english")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    emotion_labels: list[str]
    discover: Callable[[Path], list]
    split_samples: Callable[..., tuple[list, list, list]]
    default_data_dir: Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


_EMODB_SPEC = DatasetSpec(
    name="emodb",
    display_name="EmoDB",
    emotion_labels=EMODB_EMOTION_LABELS,
    discover=discover_emodb_samples,
    split_samples=emodb_split_samples,
    default_data_dir=_root() / "database" / "EmoDB",
)

DATASETS: dict[str, DatasetSpec] = {
    "casia": DatasetSpec(
        name="casia",
        display_name="CASIA",
        emotion_labels=CASIA_EMOTION_LABELS,
        discover=discover_casia_samples,
        split_samples=casia_split_samples,
        default_data_dir=_root() / "database" / "CASIA",
    ),
    "emodb": _EMODB_SPEC,
    "emo_db": _EMODB_SPEC,
    "berlin": _EMODB_SPEC,
    "english": _EMODB_SPEC,  # legacy CLI alias → EmoDB
}


def get_dataset(name: str) -> DatasetSpec:
    key = name.lower().strip()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[key]


def resolve_data_dir(dataset: str, data_dir: Path | None) -> Path:
    spec = get_dataset(dataset)
    return data_dir if data_dir is not None else spec.default_data_dir


def _canonical_dataset(dataset: str) -> str:
    key = dataset.lower().strip()
    if key in ("emodb", "emo_db", "berlin", "english"):
        return "emodb"
    return key


def default_baseline_run_name(
    dataset: str,
    seed: int,
    *,
    speaker_independent: bool,
    resplit_random: bool = False,
) -> str:
    """Run-name substring for locating baseline checkpoints."""
    _ = resplit_random
    if speaker_independent:
        return "si_s"
    return f"seed{seed}_rand"


def default_ensemble_run_suffix(
    dataset: str,
    seed: int,
    strategy: str,
    *,
    speaker_independent: bool,
    resplit_random: bool = False,
) -> str:
    """Ensemble run-name suffix for Table 2/3 result dirs."""
    _ = (dataset, resplit_random)
    if speaker_independent:
        return f"si_s{seed}_{strategy}"
    return f"seed{seed}_rand_{strategy}"


def members_json_name(dataset: str, seed: int) -> str:
    ds = _canonical_dataset(dataset)
    if ds == "casia":
        return f"members_seed{seed}.json"
    return f"members_{ds}_seed{seed}.json"


def dataset_results_dir(results_root: Path, dataset: str) -> Path:
    """Per-dataset results tree: ``results/{casia|emodb}/``.

    If ``results_root`` is already the dataset folder (name matches canonical
    dataset), it is returned unchanged.
    """
    root = Path(results_root)
    canon = _canonical_dataset(dataset)
    if root.name == canon:
        return root
    return root / canon


def member_search_roots(results_root: Path, dataset: str) -> list[Path]:
    """Directories that may contain ``{model}/<run>/`` for this dataset.

    Prefers ``results/{dataset}/``. Also checks the legacy flat ``results/`` tree;
    callers must still filter by ``dataset_key`` so CASIA/EmoDB are not mixed.
    """
    root = Path(results_root)
    canon = _canonical_dataset(dataset)
    roots: list[Path] = []
    if root.name == canon:
        roots.append(root)
        roots.append(root.parent)
    else:
        roots.append(root / canon)
        roots.append(root)
    out: list[Path] = []
    seen: set[Path] = set()
    for p in roots:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def paper_dir_for_dataset(results_root: Path, dataset: str) -> Path:
    """Paper exports under ``results/{dataset}/paper/`` (shared tables stay readable)."""
    return dataset_results_dir(results_root, dataset) / "paper"
