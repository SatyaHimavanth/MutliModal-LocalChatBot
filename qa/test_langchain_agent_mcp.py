import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src-tauri" / "Cargo.toml"

raise SystemExit(
    subprocess.run(
        [
            "cargo",
            "test",
            "--manifest-path",
            str(MANIFEST),
            "--lib",
            "langchain_runtime::tests",
        ],
        cwd=ROOT,
        check=False,
    ).returncode
)
