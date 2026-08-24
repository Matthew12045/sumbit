Repo: /home/claude/sumbit (or https://github.com/Matthew12045/sumbit), main
branch, clean (the streaming fix already merged).

Goal: add an post-processing pass that uses qwen local gateway with the kien-thai Thai-writing skill
(https://github.com/chakrit/kien-thai) to polish the prose fields of the
meeting summary before posting, running the skill's real audit+fix loop to
convergence (not a bounded single pass) since token/latency cost is not a
constraint for this feature. This audits/revises a qwen-drafted summary
rather than drafting from scratch, matching the skill's own documented
best-use case for a Thai-native draft.

Do this:

1. Clone https://github.com/chakrit/kien-thai and vendor
   skills/kien-thai/SKILL.md + skills/kien-thai/references/*.md into this
   repo under meeting_bot/thai_skill/ (keep the same filenames). Do not
   depend on fetching these from GitHub at runtime. Note the vendored
   source/commit in a short comment or README in that folder for future
   updates.

2. Build the register-scoped audit bundle for register 6 ("Official /
   minutes") as the design in references/register.md's Two-tier injection
   note describes: SKILL.md in full, all mechanical/craft/grammar/
   style-rules/forbidden-phrases references in full, but register.md and
   examples.md/exemplars.md scoped down to the shared register-agnostic
   sections + the "Register 6 — Official / minutes" section only, not all
   six registers. This scoping is about output quality (stopping the model
   from blending registers), not a token-savings measure -- keep it exactly
   as the skill designs it even though budget isn't a constraint here. Build
   this once at import/init time (cache it), not per-call.

3. New module meeting_bot/thai_polish.py (anthropic imported lazily, same
   pure-import-rule pattern as summarizer.py):

   class ThaiPolisher:
       def __init__(self, cfg: Config): ...
       def polish(self, summary: Summary) -> Summary: ...

   polish() runs the REAL kode-thai audit+fix loop: repeatedly call Claude
   with the bundle from step 2 against the current draft of summary.overview,
   each summary.topics[i].detail, and each summary.decisions[i].rationale,
   asking it to audit and revise per the kien-thai frames and register-6
   rules, until a pass reports zero edits across all three field groups.
   Cap iterations at cfg.polish_max_passes as a safety net against a
   non-converging/thrashing run (the failure mode kien-thai's own CLAUDE.md
   documents for slimmed rule bundles -- note in the docstring that this cap
   exists purely to prevent that class of bug, not to bound cost). If the
   cap is hit without a zero-edit pass, log a warning and return the
   ORIGINAL summary unmodified (do not post an unconverged/possibly-thrashed
   draft). Every other field (titles, action_items, owner, due,
   open_questions) must be passed through byte-identical across every pass
   -- do not let the model touch them even if it wants to "improve" them.

   Use structured/JSON output each pass so you can map edited text back to
   the exact fields without fuzzy matching, and so you can detect "zero
   edits" precisely (compare the returned field values to the previous
   pass's values, not just to the qwen original). On ANY failure -- API
   error, timeout, malformed response, missing/extra fields, empty output
   -- log a warning and return the ORIGINAL summary unmodified. This must
   never raise past bot.py and must never block posting.

4. Config additions (meeting_bot/config.py + .env.example, keep the pure-
   import rule intact):
   - ANTHROPIC_API_KEY -- real Anthropic API key, separate from
     ANTHROPIC_AUTH_TOKEN (which stays pointed at the qwen gateway; do not
     reuse or conflate these two).
   - POLISH_ENABLED (bool, default false) -- still needs its own
     probe-and-eyeball validation round before flipping on, same as any
     other pipeline change; unlimited budget doesn't change that.
   - POLISH_MODEL (default claude-opus-4-8) -- deliberately the more
     capable model, not sonnet, since cost is not the constraint for a
     linguistic-judgment task like this.
   - POLISH_MAX_PASSES (default 20) -- safety-net cap on loop iterations,
     framed as a bug guard against non-convergence, not a budget.
   - POLISH_TIMEOUT_SECONDS (default 120) -- per-call SDK timeout, so a
     single hung request can't block posting forever; this bounds one call
     in the loop, not the whole loop.
   Add a doctor() check: if POLISH_ENABLED, verify ANTHROPIC_API_KEY is
   set and do a cheap live probe against the real Anthropic API (not the
   gateway) -- ok/fail line, never raises, same pattern as the existing
   gateway probe.

5. Wire into bot.py: after parse_summary(), if cfg.polish_enabled, run
   ThaiPolisher(cfg).polish(summary) via asyncio.to_thread before
   build_embed()/Poster.post(). Log clearly for each meeting: whether polish
   ran, was skipped (disabled), converged (and after how many passes), hit
   the safety cap and fell back, or fell back due to a hard failure -- this
   needs to be debuggable from logs alone.

6. Write tests/test_thai_polish.py: fake-client tests covering (a) the
   field-mapping (only overview/detail/rationale touched, everything else
   passed through identically across multiple simulated passes), (b) loop
   convergence (fake client returns edits for N passes then zero edits,
   confirm it stops at pass N+1 and not before/after), (c) the
   safety-cap-without-convergence fallback (fake client always returns
   edits, confirm it stops at POLISH_MAX_PASSES and falls back to the
   original), (d) fallback on any mid-loop API failure, and (e)
   POLISH_ENABLED=false skipping the call entirely (no anthropic import
   attempted in that path -- check this explicitly, it matters for the
   pure-import rule).

7. Extend tools/e2e_summarize_probe.py (or add a new
   tools/e2e_polish_probe.py alongside it) to run the full pipeline WITH
   polish enabled against real transcript(s), printing before/after for
   each polished field side by side once the loop converges, plus how many
   passes it took to converge, total added latency, and total token usage
   summed across every pass in the loop (report these for visibility, not
   because they gate anything).

8. Do NOT set POLISH_ENABLED=true anywhere, do NOT commit real API keys,
   do NOT commit or push -- leave everything in the working tree.

Report back with:
- The register-6-scoped bundle's actual size (bytes and rough token
  estimate)
- How many passes the loop actually took to converge on 2-3 real e2e
  transcripts, plus 2-3 before/after examples (original qwen prose next to
  the final converged version) so I can judge quality myself
- Total latency and total token count across the whole converged loop for
  one real meeting summary
- Confirmation the safety-cap fallback actually fires (e.g. temporarily
  force the fake/real client to never stop editing and confirm it falls
  back to qwen's original text at POLISH_MAX_PASSES rather than looping
  forever or posting a thrashed draft)
- Confirmation the hard-failure fallback still works (e.g. temporarily
  break the API key and confirm it still posts using qwen's original text)
- Anything from the vendored skill that didn't map cleanly to a
  multi-pass, per-field JSON edit loop (the skill assumes an agent doing
  full-file edits in a loop, and this is close to that now, but it's still
  constrained to three specific fields rather than a whole file -- tell me
  what friction remained, not just that it worked)