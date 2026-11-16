import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# config keys whose names differ from the rewardbench CLI flag
ARG_RENAMES = {"attention_implementation": "attn_implementation"}


def dataset_groups(data_dir):
    # the datasets of the main evaluation are scored one file per call,
    # the length control and the template variants together, the commonsense items together
    controls = sorted(data_dir.glob("*_matched_unrelated.jsonl"))
    main = [[p] for p in sorted(data_dir.glob("*.jsonl")) if p not in controls]
    templates = sorted(p for p in data_dir.glob("t*/*.jsonl") if not p.stem.endswith("_items"))
    items = sorted(data_dir.glob("t*/*_items.jsonl"))
    return main + [controls + templates, items]


def rewardbench_command(model, cfg, batch_size):
    try:
        from transformers.utils import is_flash_attn_2_available
        flash_attn = is_flash_attn_2_available()
    except ImportError:
        flash_attn = False

    cmd = [str(Path(sys.executable).parent / "rewardbench"), f"--model={cfg.get('model', model)}"]
    for key, value in cfg.items():
        if key in ("dpo", "num_gpus", "model"):
            continue
        key = ARG_RENAMES.get(key, key)
        if key == "batch_size" and batch_size:
            value = batch_size
        # flash_attention_2 needs the flash_attn package; fall back to sdpa when it is missing
        if key == "attn_implementation" and value == "flash_attention_2" and not flash_attn:
            value = "sdpa"
        if value is True:
            cmd.append(f"--{key}")
        elif value not in (False, None):
            cmd.append(f"--{key}={value}")
    return cmd


def score(model, cmd, env, files, data_dir, output_dir):
    rows = []
    for path in files:
        name = str(path.relative_to(data_dir).with_suffix(""))
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            meta = {k: v for k, v in r.items() if k not in ("chosen", "rejected", "prompt")}
            rows.append({"chosen": r["chosen"], "rejected": r["rejected"], "prompt": r["prompt"],
                         "dataset": name, "row_id": i, "meta": json.dumps(meta, ensure_ascii=False)})

    tmp_dir = output_dir / "tmp" / model.replace("/", "__")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    bundle = tmp_dir / "pairs.json"
    bundle.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    subprocess.run(cmd + [f"--dataset={bundle}", "--load_json", f"--output_dir={tmp_dir}/"], check=True, env=env)

    scored = tmp_dir / f"{model}_outputs.jsonl"
    by_dataset = {}
    for line in scored.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        r.update(json.loads(r.pop("meta")))
        by_dataset.setdefault(r.pop("dataset"), []).append(r)
    assert sum(map(len, by_dataset.values())) == len(rows), "row count mismatch after scoring"
    for name, scored_rows in by_dataset.items():
        scored_rows.sort(key=lambda r: r["row_id"])
        out = output_dir / name / f"{model}_outputs.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows) + "\n", encoding="utf-8")
        print(f"{model} {name}: {len(scored_rows)} pairs, "
              f"chosen preferred in {sum(r['results'] for r in scored_rows) / len(scored_rows):.3f}", flush=True)
    bundle.unlink()
    scored.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="models to evaluate (default: all models in the config)")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "models.yaml")
    parser.add_argument("--data_dir", type=Path, default=Path("preference_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for model in args.models or list(config):
        cfg = config[model]
        cmd = rewardbench_command(model, cfg, args.batch_size)
        env = os.environ.copy()
        if cfg.get("dpo"):
            # the DPO inference cannot handle models sharded across GPUs, so these runs are pinned to one GPU
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        for files in dataset_groups(args.data_dir):
            try:
                score(model, cmd, env, files, args.data_dir, args.output_dir)
            except subprocess.CalledProcessError as e:
                print(f"{model} failed with exit code {e.returncode}, continuing...", flush=True)


if __name__ == "__main__":
    main()
