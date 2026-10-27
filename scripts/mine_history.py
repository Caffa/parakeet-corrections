#!/usr/bin/env python3
"""Mine Handy transcription history for recurring Parakeet mistranscriptions.

Stdlib only: sqlite3, json, argparse, difflib, re.
"""

import argparse
import csv
import difflib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DB = os.path.expanduser(
    "~/Library/Application Support/com.pais.handy/history.db"
)

VOWELS_RE = re.compile(r"[aeiou]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def privacy_notice() -> None:
    print(
        "Privacy notice: this script reads your local Handy history database only. "
        "No transcripts or counts are uploaded.\n",
        file=sys.stderr,
    )


def check_db_path(db_path: str) -> int | None:
    """Return an exit code if the DB path is unusable, otherwise None."""
    p = Path(db_path)
    if not p.exists():
        print(f"Error: history database not found: {db_path}", file=sys.stderr)
        print(
            "Upstream Handy default: ~/Library/Application Support/com.pais.handy/history.db",
            file=sys.stderr,
        )
        print(
            "pupontech/Handy-Dictionary fork users: data lives under "
            "~/Library/Application Support/com.pupontech.handy.dictionary/history.db "
            "(pass --db).",
            file=sys.stderr,
        )
        return 1
    if not p.is_file():
        print(f"Error: not a regular file: {db_path}", file=sys.stderr)
        return 1
    return None


def fetch_rows(db_path: str, columns: tuple = ("transcription_text", "ground_truth")) -> list:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        col_list = ", ".join(columns)
        cur.execute(f"SELECT {col_list} FROM transcription_history")
        rows = cur.fetchall()
    finally:
        conn.close()
    return rows


def effective_text(row: tuple) -> str:
    """Use ground_truth when present, otherwise transcription_text."""
    text = row[1] if len(row) > 1 and row[1] is not None else row[0]
    return text or ""


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def consonant_skeleton(token: str) -> str:
    return VOWELS_RE.sub("", token.lower())


def distance_ratio(a: str, b: str) -> float:
    """Normalized edit distance: 0.0 = identical, 1.0 = completely different."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 1.0 - ratio


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def cmd_rare(args: argparse.Namespace) -> int:
    err = check_db_path(args.db)
    if err is not None:
        return err
    rows = fetch_rows(args.db)
    token_counts = Counter()
    bigram_counts = Counter()
    for row in rows:
        tokens = tokenize(effective_text(row))
        token_counts.update(tokens)
        bigram_counts.update(bigrams(tokens))

    items: list[tuple[str, int]] = []
    items.extend((f"{bg[0]} {bg[1]}", c) for bg, c in bigram_counts.items() if c >= args.min_count)
    items.extend((tok, c) for tok, c in token_counts.items() if c >= args.min_count)
    items.sort(key=lambda x: (-x[1], x[0]))

    fmt = args.format
    if fmt == "handy":
        print(json.dumps([{"token": t, "count": c} for t, c in items], indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["token", "count"])
        for t, c in items:
            writer.writerow([t, c])
    else:
        print(f"# Rare tokens/bigrams (min count {args.min_count})\n")
        for t, c in items:
            print(f"{c:4d}  {t}")
    return 0


def cluster_tokens(token_counts: Counter, dist_threshold: float = 0.35) -> list[dict]:
    """Group suspect tokens by consonant skeleton + normalized edit distance."""
    skeleton_map: dict[str, list[str]] = defaultdict(list)
    for token in token_counts:
        skeleton_map[consonant_skeleton(token)].append(token)

    clusters: list[dict] = []
    used = set()
    for skeleton, tokens in skeleton_map.items():
        if len(tokens) < 2:
            continue
        # Build connected components within this skeleton group.
        graph: dict[str, set[str]] = {t: set() for t in tokens}
        for i, a in enumerate(tokens):
            for b in tokens[i + 1 :]:
                if distance_ratio(a, b) <= dist_threshold:
                    graph[a].add(b)
                    graph[b].add(a)

        visited = set()
        for start in tokens:
            if start in visited or start in used:
                continue
            component = []
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                for nxt in graph[cur]:
                    if nxt not in visited:
                        stack.append(nxt)
            if len(component) < 2:
                continue
            total = sum(token_counts[t] for t in component)
            clusters.append(
                {
                    "skeleton": skeleton,
                    "variants": sorted(component, key=lambda t: (-token_counts[t], t)),
                    "counts": {t: token_counts[t] for t in component},
                    "total_count": total,
                }
            )
            used.update(component)

    clusters.sort(key=lambda c: (-c["total_count"], c["variants"][0]))
    return clusters


def cmd_cluster(args: argparse.Namespace) -> int:
    err = check_db_path(args.db)
    if err is not None:
        return err
    rows = fetch_rows(args.db)
    token_counts = Counter()
    for row in rows:
        token_counts.update(tokenize(effective_text(row)))

    clusters = cluster_tokens(token_counts)
    fmt = args.format
    if fmt == "handy":
        print(json.dumps(clusters, indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["cluster", "variants", "total_count"])
        for c in clusters:
            writer.writerow([c["skeleton"], " ".join(c["variants"]), c["total_count"]])
    else:
        print("# Suspect token clusters (shared consonant skeleton + edit distance)\n")
        for c in clusters:
            parts = [f"{v}({c['counts'][v]})" for v in c["variants"]]
            print(f"{c['total_count']:4d}  {' '.join(parts)}")
    return 0


def load_wordlist(path: str) -> set[str]:
    words = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    return words


def load_glossary(path: str) -> list[tuple[str, list[str]]]:
    glossary = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            term = parts[0]
            synonyms = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            glossary.append((term, [s.strip() for s in synonyms if s.strip()]))
    return glossary


def find_near_matches(tokens: list[str], target: str, synonyms: list[str]) -> list[tuple[str, int, list[str]]]:
    """Return (variant, count, contexts) for tokens near the intended target.

    - Exact matches of the intended target are skipped (they are correct).
    - Exact matches of a synonym emit (synonym -> intended).
    - Near-matches of the intended term emit (variant -> intended).
    - Each token is counted at most once; results are deduped by (variant, intended).
    """
    target_l = target.lower()
    synonyms_l = [s.lower() for s in synonyms]
    counts: Counter[str] = Counter()
    contexts: dict[str, list[str]] = defaultdict(list)

    for i, tok in enumerate(tokens):
        if tok == target_l:
            continue
        # Exact synonym hit: the synonym itself is a known variant of intended.
        if tok in synonyms_l:
            counts[tok] += 1
            contexts[tok].append(i)
            continue
        # Near-match of intended or any synonym.
        for t in [target_l] + synonyms_l:
            if distance_ratio(tok, t) <= 0.35 or consonant_skeleton(tok) == consonant_skeleton(t):
                counts[tok] += 1
                contexts[tok].append(i)
                break
    return [(v, counts[v], contexts[v]) for v in counts]


def extract_context(words: list[str], idx: int, radius: int = 8) -> str:
    start = max(0, idx - radius)
    end = min(len(words), idx + radius + 1)
    return " ".join(words[start:end])


def cmd_align(args: argparse.Namespace) -> int:
    if not args.glossary:
        print("align requires --glossary", file=sys.stderr)
        return 1
    glossary_path = Path(args.glossary)
    if not glossary_path.is_file():
        print(f"Error: glossary file not found: {args.glossary}", file=sys.stderr)
        return 1
    glossary = load_glossary(args.glossary)
    err = check_db_path(args.db)
    if err is not None:
        return err
    rows = fetch_rows(args.db)
    transcripts = [effective_text(row) for row in rows]
    words_per_transcript = [TOKEN_RE.findall(t) for t in transcripts]
    tokens_per_transcript = [[w.lower() for w in words] for words in words_per_transcript]

    candidates: dict[str, dict] = {}
    wordlist = load_wordlist(args.wordlist) if args.wordlist else None

    for intended, synonyms in glossary:
        for words, tokens in zip(words_per_transcript, tokens_per_transcript):
            for variant, count, indices in find_near_matches(tokens, intended, synonyms):
                key = (variant, intended)
                if key not in candidates:
                    candidates[key] = {
                        "mistranslation": variant,
                        "correction": intended,
                        "count": 0,
                        "contexts": [],
                    }
                cand = candidates[key]
                cand["count"] += count
                for idx in indices:
                    ctx = extract_context(words, idx, radius=8)
                    if ctx not in cand["contexts"]:
                        cand["contexts"].append(ctx)

    results = sorted(candidates.values(), key=lambda c: -c["count"])

    if wordlist:
        for cand in results:
            cand["is_real_word"] = cand["mistranslation"].lower() in wordlist

    if args.no_context:
        for cand in results:
            cand.pop("contexts", None)

    fmt = args.format
    if fmt == "handy":
        out = [{"mistranslation": c["mistranslation"], "correction": c["correction"]} for c in results]
        print(json.dumps(out, indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["mistranslation", "correction", "count"])
        for c in results:
            writer.writerow([c["mistranslation"], c["correction"], c["count"]])
    else:
        print("# Glossary alignment candidates\n")
        for c in results:
            extra = ""
            if "is_real_word" in c:
                extra = f" [real_word={c['is_real_word']}]"
            print(f"{c['count']:4d}  {c['mistranslation']:20s} -> {c['correction']}{extra}")
            if not args.no_context:
                for ctx in c.get("contexts", [])[:3]:
                    print(f"        ... {ctx} ...")
    return 0


def build_seed_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE transcription_history ("
        "id INTEGER PRIMARY KEY, "
        "transcription_text TEXT, "
        "ground_truth TEXT"
        ")"
    )
    seed = [
        "Olama and open code are my daily tools",
        "open code is great but Olama is faster",
        "I use quenn sometimes and supstack for writing",
        "kemaki kamaki kumaki are all misheard",
        "the thread I read was about tread",
        "treads thread and tread threads are confusing",
        "zetel zetal zatal zethel all mean zettel",
        "Hogwoods slitterin and hufflepath fans",
        "transmigated and fulfiled are misspellings",
    ]
    for i, text in enumerate(seed, 1):
        conn.execute(
            "INSERT INTO transcription_history (id, transcription_text, ground_truth) VALUES (?, ?, ?)",
            (i, text, None),
        )
    conn.commit()
    return conn


def selftest() -> int:
    """Run all three modes against an in-memory seed DB and report PASS/FAIL."""
    conn = build_seed_db()
    # sqlite3 connection has no path in memory; we pass a path string for fetch_rows,
    # so write a temp file for the in-memory test.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_fd)
    try:
        disk_conn = sqlite3.connect(tmp_path)
        conn.backup(disk_conn)
        disk_conn.close()

        class Args:
            db = tmp_path
            min_count = 1
            format = "markdown-report"
            glossary = None
            no_context = True
            wordlist = None

        args = Args()
        results = {}

        def capture(func, test_args):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                func(test_args)
                return sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

        # rare
        try:
            rare_out = capture(cmd_rare, args)
            results["rare"] = "olama" in rare_out.lower() and "open code" in rare_out.lower()
            if not results["rare"]:
                print(f"rare output:\n{rare_out}", file=sys.stderr)
        except Exception as exc:
            results["rare"] = False
            print(f"rare exception: {exc}", file=sys.stderr)

        # cluster
        try:
            cluster_out = capture(cmd_cluster, args)
            lo = cluster_out.lower()
            results["cluster"] = "kemaki" in lo and "kamaki" in lo
            if not results["cluster"]:
                print(f"cluster output:\n{cluster_out}", file=sys.stderr)
        except Exception as exc:
            results["cluster"] = False
            print(f"cluster exception: {exc}", file=sys.stderr)

        # align
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as gf:
            gf.write("Ollama\t\n")
            gf.write("OpenCode\t\n")
            gf.write("thread\ttread\n")
            gf.write("zettel\tzetel,zetal\n")
            glossary_path = gf.name

        args.glossary = glossary_path
        try:
            align_out = capture(cmd_align, args)
            lo = align_out.lower()
            results["align"] = (
                "olama" in lo
                and "ollama" in lo
                and "tread" in lo
                and "thread" in lo
                and "zetel" in lo
                and "zetal" in lo
            )
            if not results["align"]:
                print(f"align output:\n{align_out}", file=sys.stderr)
        except Exception as exc:
            results["align"] = False
            print(f"align exception: {exc}", file=sys.stderr)
        finally:
            os.unlink(glossary_path)

        all_pass = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{name}: {status}")
            all_pass = all_pass and passed
        return 0 if all_pass else 1
    finally:
        conn.close()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine Handy transcription history for recurring Parakeet errors."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to history.db")
    parser.add_argument(
        "--format",
        choices=["handy", "csv", "markdown-report"],
        default="markdown-report",
        help="Output format",
    )
    subparsers = parser.add_subparsers(dest="command")

    rare_parser = subparsers.add_parser("rare", help="Token + bigram frequency sweep")
    rare_parser.add_argument(
        "--min-count", type=int, default=2, help="Minimum frequency to report"
    )
    rare_parser.add_argument("--db", default=argparse.SUPPRESS, help="Path to history.db")
    rare_parser.add_argument(
        "--format",
        choices=["handy", "csv", "markdown-report"],
        default=argparse.SUPPRESS,
        help="Output format",
    )

    cluster_parser = subparsers.add_parser(
        "cluster", help="Cluster suspect variants by edit distance and consonants"
    )
    cluster_parser.add_argument("--db", default=argparse.SUPPRESS, help="Path to history.db")
    cluster_parser.add_argument(
        "--format",
        choices=["handy", "csv", "markdown-report"],
        default=argparse.SUPPRESS,
        help="Output format",
    )

    align_parser = subparsers.add_parser(
        "align", help="Align transcripts against a glossary"
    )
    align_parser.add_argument("--glossary", required=True, help="TSV glossary file")
    align_parser.add_argument(
        "--no-context", action="store_true", help="Omit surrounding context"
    )
    align_parser.add_argument(
        "--wordlist",
        help="Path to a wordlist for collision checking (one word per line)",
    )
    align_parser.add_argument("--db", default=argparse.SUPPRESS, help="Path to history.db")
    align_parser.add_argument(
        "--format",
        choices=["handy", "csv", "markdown-report"],
        default=argparse.SUPPRESS,
        help="Output format",
    )

    parser.add_argument(
        "--selftest", action="store_true", help="Run end-to-end self-test"
    )

    args = parser.parse_args()

    if args.selftest:
        return selftest()

    privacy_notice()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "rare":
        return cmd_rare(args)
    elif args.command == "cluster":
        return cmd_cluster(args)
    elif args.command == "align":
        return cmd_align(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
