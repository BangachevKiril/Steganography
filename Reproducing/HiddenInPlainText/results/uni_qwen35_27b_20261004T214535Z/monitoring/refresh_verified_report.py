"""Regenerate the frozen report, then correct its legacy hardware label locally."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

results = Path(__file__).resolve().parents[1]
root = results.parents[1]
launch = json.loads((root / "launches/university_qwen_latest.json").read_text())
source = Path(launch["local_source"])
update = int(sys.argv[1])
assert results == Path(launch["local_results"]) and update <= launch["verified_update"]
config = json.loads((source / "configs/university_training.json").read_text())
for name, digest in config["source_sha256"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
subprocess.run([sys.executable, "-m", "stego.university_training_report", str(results),
                "--completed-update", str(update)], cwd=source, check=True,
               stdout=subprocess.DEVNULL)
report = results / "report.md"
text = report.read_text()
old = "This separately initialized L40S experiment"
new = "This separately initialized H100 experiment with the Qwen3.5-27B-FP8 observer"
assert text.count(old) == 1
text = text.replace(old, new)
note = ("\n## Monitoring provenance\n\nThe frozen report generator contains a legacy L40S hardware label; "
        "only that label was corrected in this generated report to identify the actual H100/Qwen run. "
        "The frozen source and configuration are unchanged.\n")
for audit_path in sorted((results / "monitoring").glob("EVALUATION_??????_VERIFICATION.json")):
    audit = json.loads(audit_path.read_text())
    evaluation_update = int(audit_path.name.split("_")[1])
    if evaluation_update <= update and audit.get("checkpoint_state_independently_hashed_remotely") is False:
        note += (f"\nEvaluation {audit['evaluation_update']}: "
                 f"{audit['checkpoint_state_verification_qualification']} "
                 f"See monitoring/{audit_path.name} for the full audit.\n")
report.write_text(text + note)
record = dict(updated_at=datetime.now(timezone.utc).isoformat(), completed_update=update,
              source_generator_unchanged=True, correction_from=old, correction_to=new, note=note,
              report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
              refresh_command=f"python3 {Path(__file__).resolve()} VERIFIED_UPDATE")
(results / "monitoring/REPORT_PRESENTATION_CORRECTION.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
