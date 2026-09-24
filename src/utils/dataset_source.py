"""Resolve and download dataset files from config.yaml → dataset.sources."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.helpers import c, p, t

_DEFAULT_RETRIES = 5
_CHUNK = 1024 * 256


def get_dataset_block(config) -> Dict[str, Any]:
    """Return the top-level ``dataset`` mapping from Config.extra."""
    block = config.extra.get("dataset")
    if not isinstance(block, dict):
        raise KeyError("config.yaml is missing a 'dataset' section")
    return block


def get_active_source(config) -> Dict[str, Any]:
    """
    Return the active dataset source dict (name, catalog, files, ...).

    ``dataset.active`` must match a key under ``dataset.sources``.
    """
    block = get_dataset_block(config)
    active = block.get("active")
    sources = block.get("sources") or {}
    if not active or active not in sources:
        raise KeyError(
            f"dataset.active={active!r} not found in dataset.sources "
            f"(available: {sorted(sources)})"
        )
    source = sources[active]
    if not isinstance(source, dict):
        raise TypeError(f"dataset.sources.{active} must be a mapping")
    return {"key": active, **source}


def resolve_download_dir(config, source: Optional[Dict[str, Any]] = None) -> Path:
    """Absolute download directory for the active (or given) source."""
    source = source or get_active_source(config)
    rel = source.get("download_dir") or "data"
    path = Path(rel)
    if not path.is_absolute():
        path = Path(config.paths.root) / path
    return path.resolve()


def list_download_files(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize ``files`` entries to dicts with key, filename, url."""
    files = source.get("files") or []
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(files):
        if not isinstance(item, dict):
            raise TypeError(f"dataset file entry #{i} must be a mapping")
        filename = item.get("filename")
        if not filename:
            raise ValueError(f"dataset file entry #{i} is missing 'filename'")
        out.append(
            {
                "key": item.get("key") or Path(filename).stem,
                "filename": filename,
                "url": item.get("url"),
            }
        )
    return out


def _fetch(url: str, dest: Path, *, retries: int = _DEFAULT_RETRIES) -> None:
    """Download ``url`` to ``dest`` with retries (handles truncated Zenodo transfers)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "cap6415-tree-canopy-detection/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                expected = resp.headers.get("Content-Length")
                expected_n = int(expected) if expected and expected.isdigit() else None
                got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
            if expected_n is not None and got != expected_n:
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got only {got} out of {expected_n} bytes",
                    (got, expected_n),
                )
            if got == 0:
                raise OSError("downloaded 0 bytes")
            tmp.replace(dest)
            return
        except (
            urllib.error.ContentTooShortError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as e:
            last_err = e
            p(
                "retry",
                f"{dest.name} attempt {attempt}/{retries}: {e}",
                color1=c.ORANGE,
            )
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 20))

    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_err


def download_dataset(
    config,
    *,
    force: bool = False,
    source_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download files for ``dataset.active`` (or ``source_key`` override).

    Reads URLs from config.yaml. Skips existing files unless ``force``.
    Entries with ``url: null`` are reported as manual (competition mode).
    """
    block = get_dataset_block(config)
    if source_key:
        sources = block.get("sources") or {}
        if source_key not in sources:
            raise KeyError(
                f"Unknown source {source_key!r}; choose from {sorted(sources)}"
            )
        config.extra["dataset"] = {**block, "active": source_key}

    source = get_active_source(config)
    dest_dir = resolve_download_dir(config, source)
    files = list_download_files(source)

    t("Dataset download")
    p("Active source", f"{source['key']} — {source.get('name', '')}")
    p("Catalog", source.get("catalog", "(none)"))
    p("Download dir", dest_dir)
    notes = source.get("notes")
    if notes:
        p("Notes", str(notes).strip())

    dest_dir.mkdir(parents=True, exist_ok=True)
    manual: List[str] = []
    downloaded = 0
    skipped = 0

    for entry in files:
        dest = dest_dir / entry["filename"]
        url = entry.get("url")
        partial = dest.with_suffix(dest.suffix + ".partial")

        # Drop leftover truncated downloads from a previous failed run
        if partial.exists():
            p("removing partial", partial.name, color1=c.ORANGE)
            partial.unlink()

        if dest.exists() and not force:
            p("skip (exists)", dest.name, color1=c.GREEN)
            skipped += 1
            continue

        if not url:
            manual.append(entry["filename"])
            p("manual", f"{entry['filename']} (no url in config)", color1=c.ORANGE)
            continue

        p("downloading", dest.name)
        _fetch(str(url), dest)
        p("saved", dest.name, color1=c.GREEN)
        downloaded += 1

    if manual:
        p("Place these files under", dest_dir, color1=c.ORANGE)
        for name in manual:
            p("", str(dest_dir / name))
        p("Catalog", source.get("catalog"))

    summary = {
        "source": source["key"],
        "download_dir": dest_dir,
        "downloaded": downloaded,
        "skipped": skipped,
        "manual": manual,
    }
    p(
        "Done",
        f"downloaded={downloaded} skipped={skipped} manual={len(manual)}",
        color1=c.GREEN,
    )
    return summary
