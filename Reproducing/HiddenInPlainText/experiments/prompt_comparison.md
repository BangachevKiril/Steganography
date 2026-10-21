# Original versus revised sender prompt

This frozen-model diagnostic compares the original sender system prompt with
the revised prompt approved in the conversation. It uses vanilla Qwen3-4B with
no adapters or updates, on the exact 100 training questions from the vanilla
rejection baseline. For each question, both bit values are answered under each
prompt: 200 answers per prompt, 400 total.

Communicator identity is fixed to sender A, receiver B. One fresh public 16-bit
string is shared across the four answers for a question. Initial generation
and judge seeds also match across those four cells. Every row has an independent
random-number stream so another row's generation length does not affect it.
Both arms use the same batched sampler and unchanged thinking transition.

Thinking budget: 256 tokens, with the original forced closing token at the cap.
Final answer budget: 128 tokens. Temperature: 0.6, full-vocabulary sampling.
The judge prompt, constrained PASS/FAIL output, and separate format check are
unchanged. The same frozen base weights serve answerer and judge with fresh
contexts. The judge receives only Q and the public answer.

Primary reported metrics are judge rejection, format failure, their overlap,
combined rejection, final-answer truncation, and forced thinking termination.
Additional literal text flags report mentions of a bit value, mentions of the
hidden bit, exact copies of the public string, and discussion of the public
string. These are transparent diagnostic proxies, not learned detectors or
proof of communication success. They do not affect generation or rewards.
Natural numbers in practical answers are not treated as bit disclosure merely
because they are 0 or 1. Public answers are preserved verbatim.

Report rates per 200 answers while identifying the 100 distinct question units;
the two bit conditions for a question are related observations. Include paired
improvement/worsening counts and inspect examples to interpret text flags.
No receiver or interceptor is run, so this diagnostic makes no claim about
hidden-channel reliability or secrecy.

Use a separate immutable release and run directory. Verify source/data/reference
artifact hashes before inference. Save answers before judgment and resume only
from matching committed batches. Keep private reasoning artifacts remote and
out of displayed results. No previous experiment is modified or rerun.
