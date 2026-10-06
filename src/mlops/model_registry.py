from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

BUNDLE_FILES = [
    "failure_model.joblib",
    "anomaly_model.joblib",
    "rul_model.joblib",
    "streaming_metadata.json",
    "drift_baseline.json",
]


def register_model_bundle(model_dir: Path, metadata: dict, registry_dir: Path = Path("models/registry")) -> str:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = registry_dir / "versions" / version
    version_dir.mkdir(parents=True, exist_ok=True)

    for filename in BUNDLE_FILES:
        source = model_dir / filename
        if source.exists():
            shutil.copy2(source, version_dir / filename)

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metadata,
        "files": [name for name in BUNDLE_FILES if (version_dir / name).exists()],
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "champion.json").write_text(
        json.dumps({"version": version, "path": str(version_dir)}, indent=2)
    )

    metadata_path = model_dir / "streaming_metadata.json"
    if metadata_path.exists():
        current = json.loads(metadata_path.read_text())
        current["registry_version"] = version
        metadata_path.write_text(json.dumps(current, indent=2))
    return version
