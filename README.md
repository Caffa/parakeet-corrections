# parakeet-corrections

Ready-to-import correction lexicons and a mining tool for Handy's Parakeet transcriptions.

Parakeet wrote "Olama" 49 times and "open code" 55 times in one user's first four months of real dictation. This repo is what that looked like fixed: a small, importable set of deterministic post-processing replacements, mined from 14,941 transcriptions and roughly 320,000 words between 2026-06 and 2026-10 (as of 2026-10-27).

## TL;DR

- Copy `data/all.json` (or `formats/vocabulary/all.txt`) into Handy via the third-party [Handy-Dictionary](https://github.com/pupontech/Handy-Dictionary) fork, or add rules manually upstream.
- Run `scripts/mine_history.py` against your own `history.db` to grow the list from your own recurring mistranscriptions.
- These are deterministic post-processing rewrites, not model fixes. They will not change what Parakeet hears; they only patch what Handy pastes.

## Why Parakeet gets these wrong

Parakeet is an NVIDIA Parakeet TDT (Token-and-Duration Transducer) model. It is fast, accurate, and very small on disk, which is why Handy adopted it. Its weakness is the same weakness every transducer has: greedy decoding with no language-model rescoring and, in Handy's case, no runtime vocabulary or hotword hook. The open-source inference path Handy uses, [`transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp), exposes no word-boosting API for Parakeet. There is a standing feature request: [transcribe.cpp#126](https://github.com/handy-computer/transcribe.cpp/issues/126). NVIDIA's own NIM documentation does describe word boosting for Parakeet TDT and RNN-T models, but that API is not available in the local runtime Handy ships.

The practical result is that rare proper nouns, brand names, supplement terms, and fandom vocabulary get mangled into phonetic neighbors that the transducer is more comfortable emitting. OpenCode becomes "open code" because the model has no concept that a camel-cased product name exists. Ollama becomes "Olama" because that vowel is acoustically close and statistically cheaper. Zettel becomes a family of variants because the model has rarely seen the word in training data.

There is an interesting project to watch: [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio) is an active Swift/CoreML implementation of Parakeet. It is worth monitoring because any vocabulary or hotword support it gains may eventually reach macOS inference paths. This repo does not depend on that, and it does not claim those decoder hooks exist today.

## What you can fix today vs not

You can fix deterministic, recurring token-level mistakes. If Parakeet emits the exact same wrong string every time you say a rare word, a replacement rule is a clean bandage.

You cannot fix one-off hallucinations, context-dependent errors, or homophone collisions. You cannot make the model more likely to emit "OpenCode" at decode time. You also cannot safely rewrite words that have legitimate alternative meanings: "tread" really is a word, and "Asian" really is a word. Adding those blindly would corrupt correct text. Those cases stay personal and stay out of `all.json`.

## Install into Handy

Upstream Handy has no bulk-import button. The honest paths are:

1. **Bulk import via Handy-Dictionary.** The third-party fork at [pupontech/Handy-Dictionary](https://github.com/pupontech/Handy-Dictionary) adds import/export for Word Replacements. Import the generated `formats/handy-dictionary/dictionary.json` (verified schema) into the fork, or copy rules manually. See `formats/handy-dictionary/README.md` for the schema details and the pending CSV status.

2. **Manual upstream path.** Open Handy, go to Settings → Accuracy → Word Replacements, set Word Correction Mode to **Exact Replacements**. The default is "Prefer Custom Words," which is a trap: it will sometimes keep the model's guess when you want a hard replacement. Then add each mistranslation→correction pair by hand.

3. **Power-user batch path.** Quit Handy, then batch-edit the `word_replacements` array in `~/Library/Application Support/com.pais.handy/settings_store.json` with a text editor or small script. `data/all.json` matches that array's shape exactly. Restart Handy.

`formats/vocabulary/all.txt` is a plain corrections/custom-words list. In the fork it acts as a vocabulary bias (custom words), not replacement pairs. That bias currently only affects Whisper models; it is inert for Parakeet in Handy's runtime, so it is useful to Whisper users but does not change Parakeet output.

Engine facts, taken from Handy's code: matching is case-insensitive and word-boundary aware. Multi-word rules are supported. There is no plural or inflection handling, so plurals must be added as separate rules if you need them. The correction inherits the input token's case pattern: all-caps input → all-caps correction; leading-capital input → capitalised correction; otherwise the correction is emitted verbatim (so a lowercase input such as "open code" produces "OpenCode").

## Self-mining methodology

The seed data came from one user's local `history.db`, usually at:

```
~/Library/Application Support/com.pais.handy/history.db
```

Users of the pupontech/Handy-Dictionary fork have history under `~/Library/Application Support/com.pupontech.handy.dictionary/history.db` and must pass `--db` to mine it.

The table is `transcription_history`, and the relevant columns are `transcription_text` and `ground_truth`. When `ground_truth` is non-null, the corrected value is treated as the authoritative text for counting.

The mining tool has three subcommands:

- `rare` — token and bigram frequency sweep. Use `--min-count` to filter. Example:
  ```bash
  python3 scripts/mine_history.py rare --min-count 3
  ```
- `cluster` — flag suspect tokens and group phonetic variants by normalized edit distance plus a shared consonant-skeleton heuristic. Example:
  ```bash
  python3 scripts/mine_history.py cluster
  ```
- `align` — point a glossary at the transcripts and emit candidate correction pairs. Example:
  ```bash
  python3 scripts/mine_history.py align --glossary my_glossary.tsv --wordlist /usr/share/dict/words
  ```

The glossary file is a TSV where each line is `intended_term` followed by an optional tab-separated comma list of synonyms:

```
Ollama
OpenCode
zettel	zetel,zetal
```

By default, `align` prints roughly ±8 words of context around each hit to stdout. Use `--no-context` to emit only the candidate pairs. Use `--wordlist FILE` to flag variants that are already real English words, which is usually a signal to keep the rule personal. The flag only needs to be present when you want the collision check.

Emit JSON that downstream tools consume with `--format handy` (for `align`, this is a list of `{"mistranslation", "correction"}` objects; for `rare` and `cluster` it is the plain JSON of their native token/cluster records). Example:

```bash
python3 scripts/mine_history.py align --glossary my_glossary.tsv --wordlist /usr/share/dict/words --format handy
```

Worked example counts from one user's log (as of 2026-10-27):

| Mistranscription(s) | Correction | Count |
| --- | --- | --- |
| open code | OpenCode | 55 |
| Olama | Ollama | 49 |
| kemaki / kamaki / kumaki | Kimaki | 29 |
| tread | thread | 24 |
| Asian / Asians | agent(s) | 25 |
| zetel / zetal / zatal / zethel | zettel | 20 |
| Hogwoods | Hogwarts | 11 |

The `Asian → agent` pair is the canonical risky case: both words are real, both are socially loaded, and a context-free rewriter would corrupt genuine uses. That is why it, and other collision cases, are kept out of the shared safe lexicon.

## The collision problem

`data/risky.json` holds entries that are too dangerous to ship to everyone. Each entry has four fields: `mistranslation`, `correction`, `collides_with`, and `note`. The `collides_with` field names the legitimate word that would be overwritten, and `note` explains the tradeoff.

`data/all.json` deliberately excludes these. If you decide to import `risky.json` into your own Handy instance, you are accepting that you will occasionally rewrite real words. That is a personal call, not a default.

## Contributing (v1)

This is version 1 of the lexicon. For now, contributions are issue-only. Open an issue and include a Markdown or JSON block with the mistranslation, correction, observed count, and approximate context type. Mined pairs with counts are preferred over invented ones. Please do not submit personal names or sensitive information.

## Privacy & provenance

The seed data is one real user's dictation log. All entries have been reduced to generic product, brand, supplement, food, or fandom names. No personal names appear. No transcripts are included. No medical conditions, dosages, or identifying contexts are exposed. The published terms are generic over-the-counter, supplement, or food vocabulary, plus a few software and media proper nouns.

## Roadmap

- Grow the safe lexicon from contributors' mined counts.
- Add per-profile variant support so risky collision rules can be opted into safely.
- Watch [transcribe.cpp#126](https://github.com/handy-computer/transcribe.cpp/issues/126). The day a runtime vocabulary or word-boosting hook lands, lexicon entries can graduate from post-hoc replacements into decoder hints.

## License

MIT License. See `LICENSE`. Attribution is appreciated but not required.
