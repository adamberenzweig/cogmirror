"""Generate the M3 story set and its narration audio.

    .venv/bin/python scripts/generate_stories.py                 # stories + audio, skipping what exists
    .venv/bin/python scripts/generate_stories.py --plan          # print the plan and exit (no API calls)
    .venv/bin/python scripts/generate_stories.py --audio-only    # (re)render narration for existing stories
    .venv/bin/python scripts/generate_stories.py --verify        # re-run the content checks on stories.jsonl

The LLM is chosen by whichever key is present: ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN),
GEMINI_API_KEY / GOOGLE_API_KEY, OPENAI_API_KEY; a `.env` file in the repo root is read too.
Raw responses are cached under data/stories/raw, so reruns only pay for missing stories.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogmirror import stories as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "stories"))
    ap.add_argument("--provider", choices=["anthropic", "gemini", "openai"], help="default: whichever key is set")
    ap.add_argument("--model", help=f"default per provider: {S.DEFAULT_MODELS}")
    ap.add_argument("--candidates", type=int, default=3, help="stories requested per call; the first that passes is kept")
    ap.add_argument("--only", help="comma-separated story ids to (re)generate")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--verify", action="store_true", help="check stories.jsonl and exit")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--rate", type=int, default=170, help="words per minute for `say`")
    ap.add_argument("--force-audio", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    S.load_dotenv(ROOT / ".env")

    if args.plan:
        for s in S.build_plan():
            print(f"{s.id} {s.kind:8s} {'/'.join(s.emotions) or '-':22s} {s.topic}")
        return 0

    if args.verify or args.audio_only:
        records = S.read_jsonl(root / "stories.jsonl")
        if not records:
            print("no stories yet", file=sys.stderr)
            return 1
        problems = S.verify(records)
        print(f"{len(records)} stories; {len(problems)} problems")
        for p in problems:
            print("  ", p)
        if args.audio_only:
            S.render_audio(root, records, voice=args.voice, rate=args.rate, force=args.force_audio)
        write_readme(root, records)
        return 1 if problems and args.verify else 0

    try:
        llm = S.LLM(provider=args.provider, model=args.model)
        print(f"provider={llm.provider} model={llm.model} root={root}")
    except RuntimeError as e:
        # No key: still fine if every story has a cached raw response under raw/.
        llm = None
        print(f"{e}; using cached raw responses only (root={root})")
    only = set(args.only.split(",")) if args.only else None
    records = S.generate(root, llm, n_stories=args.candidates, only=only)
    plan = S.build_plan()
    missing = [s.id for s in plan if s.id not in {r['id'] for r in records}]
    print(f"\n{len(records)}/{len(plan)} stories; usage {llm.usage if llm else 'none (cached)'}")
    if missing:
        print("missing:", ", ".join(missing))
    kinds = Counter(r["kind"] for r in records)
    print("kinds:", dict(kinds))
    problems = S.verify(records)
    if problems:
        print("problems:")
        for p in problems:
            print("  ", p)
    if not args.no_audio and records:
        S.render_audio(root, records, voice=args.voice, rate=args.rate, force=args.force_audio)
    write_readme(root, records)
    return 0 if not missing and not problems else 1


def write_readme(root: Path, records: list[dict]) -> None:
    n_turn = sum(r["kind"] == "turn" for r in records)
    n_neutral = sum(r["kind"] == "neutral" for r in records)
    emotions = Counter(e for r in records for e in r["emotions"])
    sources = Counter(f"{r.get('provider')}/{r.get('model')}" for r in records)
    lines = [
        "# M3 story set",
        "",
        f"{len(records)} stories generated with the story prompt from the appendix of Anthropic's emotion-concepts paper",
        "(see `src/cogmirror/stories.py` for the exact prompts): "
        f"{len(records) - n_turn - n_neutral} single-emotion stories, {n_turn} with a scripted emotional turn "
        f"(two target emotions, turn point recorded), {n_neutral} neutral passages in the paper's neutral style as prose.",
        "Each story is 120-180 words and never names its target emotion (checked with `--verify`).",
        "The released gemotions stories were not reused.",
        "Generated by (provider/model, story count): " + ", ".join(f"{k} {n}" for k, n in sources.items()) + ".",
        "`claude-code-session` means the text was written by the Claude Code session itself, following the same prompt,",
        "because no API key was available; those responses are kept under `raw/` like any other.",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `plan.json` | the deterministic plan: id, kind, target emotion(s), topic |",
        "| `stories.jsonl` | one record per story: `id`, `kind`, `emotions`, `clusters`, `quadrants`, `topic`, `text`, `sentences`, `sentence_word_offsets`, `word_count`, `turn` (`sentence_idx`, `word_idx`), provider/model |",
        "| `text/<id>.txt` | plain text for the teleprompter |",
        "| `audio/<id>/<k>.m4a` | narration of sentence k (macOS `say`), with `index.json` listing durations |",
        "| `raw/` | raw LLM responses (cache; safe to delete) |",
        "",
        f"Target emotions ({len(emotions)}), with story counts: "
        + ", ".join(f"{e} {n}" for e, n in sorted(emotions.items())),
        "",
        "Regenerate or extend with `scripts/generate_stories.py` (idempotent).",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
