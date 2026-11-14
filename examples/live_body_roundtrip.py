"""Synthetic full-body FK -> separate headset gateway -> shadow -> A/B/C feedback.

Run with this repository's Python and --headset-python pointing at the separate
myosuite-headset environment. --out-dir must be new; retains all evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque
from pathlib import Path

from myosuite_shadow.body_cli import initialize, synthetic_stream
from myosuite_shadow.body_server import BodyServer
from myosuite_shadow.tracking_config import load_tracking_setup


async def run(headset_python: Path, headset_repo: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    config = initialize(out_dir / "model", source="synthetic")
    results = []
    for condition in "ABC":
        server = BodyServer(
            load_tracking_setup(config),
            log_dir=out_dir / f"shadow-{condition}",
        )
        await server.start(port=0)
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                str(headset_python),
                "-u",
                "-m",
                "fatigue_feedback",
                "serve-shadow",
                "--shadow-uri",
                server.uri,
                "--port",
                "0",
                "--condition",
                condition,
                "--interval-s",
                ".5",
                "--no-webxr",
                "--provider",
                "demo",
                "--log-dir",
                str(out_dir / f"gateway-{condition}"),
                cwd=str(headset_repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            startup = deque(maxlen=20)
            with (out_dir / f"gateway-{condition}-startup.log").open(
                "x", encoding="utf-8"
            ) as diagnostics:
                async with asyncio.timeout(15):
                    while True:
                        line = (await process.stdout.readline()).decode(
                            errors="replace"
                        )
                        if not line:
                            raise RuntimeError(
                                "gateway exited before listening:\n" + "".join(startup)
                            )
                        startup.append(line)
                        diagnostics.write(line)
                        diagnostics.flush()
                        if "Full-body gateway listening at " in line:
                            uri = line.split("listening at ", 1)[1].split(";", 1)[0]
                            break
            await synthetic_stream(config, uri, seconds=6.0)
            # Let the gateway flush the disconnect and feedback outcome.
            await asyncio.sleep(0.2)
            records = [
                json.loads(line)
                for line in (out_dir / f"gateway-{condition}/fullbody.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            states = [
                r["data"]["packet"]
                for r in records
                if r["type"] == "headset_output"
                and r["data"]["packet"].get("type") == "shadow_state"
            ]
            fresh = [s for s in states if s["quality"]["status"] == "running"]
            assert fresh, f"no fresh state in condition {condition}"
            for state in states:
                for region in state["regions"]:
                    assert ("activation" in region) == (condition in "BC")
                    assert ("fatigue" in region) == (condition == "C")
            if condition in "BC":
                assert any(
                    any(r["activation"] is not None for r in s["regions"])
                    for s in fresh
                )
            if condition == "C":
                assert any(
                    any(r["fatigue"] is not None for r in s["regions"]) for s in fresh
                )
            requests = [r["data"] for r in records if r["type"] == "llm_request"]
            assert requests, "missing exact feedback request log"
            for request in requests:
                assert ("loading" in request["view"]) == (condition in "BC")
                assert ("fatigue" in request["view"]) == (condition == "C")
            assert any(r["type"] == "llm_response" for r in records)
            results.append(
                {
                    "condition": condition,
                    "fresh_states": len(fresh),
                    "feedback_requests": len(requests),
                    "model_sha256": fresh[-1]["model_sha256"],
                    "capture_age_verified": fresh[-1]["quality"][
                        "capture_age_verified"
                    ],
                }
            )
            print(json.dumps(results[-1]), flush=True)
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            await server.close()
    (out_dir / "roundtrip.json").write_text(
        json.dumps(
            {"evidence": "synthetic model, separate processes", "results": results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headset-python", type=Path, required=True)
    parser.add_argument("--headset-repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(
        run(
            # Preserve the venv executable symlink on Unix: resolving it would
            # launch base Python and lose the separately installed gateway.
            args.headset_python.absolute(),
            args.headset_repo.resolve(),
            args.out_dir.resolve(),
        )
    )


if __name__ == "__main__":
    main()
