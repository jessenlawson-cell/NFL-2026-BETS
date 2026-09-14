import hashlib
import json
import sqlite3
from pathlib import Path

root = Path("/workspace")
report = json.loads((root / "reports/model_1.1.2_development.json").read_text())
manifest = json.loads((root / "manifests/model_1.1.2_development.json").read_text())
metadata_path = root / "artifacts/models/1.1.2/metadata.json"
metadata = json.loads(metadata_path.read_text())
connection = sqlite3.connect(root / "data/runtime/nfl_bets.sqlite3")
history = connection.execute(
    "SELECT model_version,command,status FROM model_history "
    "WHERE model_version LIKE '1.1.%' ORDER BY created_at_utc"
).fetchall()
counts = {
    "model_test_registry": connection.execute(
        "SELECT COUNT(*) FROM model_test_registry WHERE model_version LIKE '1.1.%'"
    ).fetchone()[0],
    "api_requests": connection.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0],
    "bet_log": connection.execute("SELECT COUNT(*) FROM bet_log").fetchone()[0],
}
connection.close()
output = {
    "status": report["status"],
    "selected_configuration": report["selected_configuration"],
    "metrics": report["development_metrics"],
    "bootstrap": report["bootstrap"],
    "prospective_start_utc": report["prospective_start_utc"],
    "artifact_hash_matches": manifest["artifact_hash"]
    == hashlib.sha256((root / manifest["artifact"]).read_bytes()).hexdigest(),
    "metadata_hash_matches": manifest["metadata_hash"]
    == hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    "spec_hash_matches": metadata["spec_hash"]
    == hashlib.sha256((root / "MODEL_SPEC_V1_1.md").read_bytes()).hexdigest(),
    "history": history,
    "counts": counts,
}
print(json.dumps(output, indent=2))
