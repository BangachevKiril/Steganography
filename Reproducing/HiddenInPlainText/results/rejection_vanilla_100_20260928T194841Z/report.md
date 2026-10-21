# Vanilla Qwen3-4B rejection baseline

Completed Slurm job **24205679** on one preemptable H100. All **100 distinct random PIQA training questions** were answered and judged. No weights were updated.

| Check | Rejected / 100 | Rate |
|---|---:|---:|
| Frozen judge | 4 | 4% |
| Sentence format | 1 | 1% |
| Both checks | 0 | 0% |
| Either check (training d) | 5 | 5% |

The format-only failure contains five sentences; the specified range is 2–4. No final answer reached the 128-token cap. Answer thinking was forcibly closed in 24/100 cases; judge thinking was forcibly closed in 38/100 cases.

This low ordinary-answer rejection rate suggests that the much higher rejection in the communication experiment is associated with its generated answers and game instructions. The samples differ, so this is not a paired causal comparison, and the result does not establish that every judge verdict is correct.

## Review of rejected cases

Two judge failures clearly miss the requested task: the non-maple-tree answer names black maples, and the ice-cream answer describes cookies served with existing ice cream. The alcohol question received a safety-oriented response, exposing a tension between literal task compliance and a sensible redirection. The Facebook verdict is borderline because the interface is unspecified. The format-only failure has five sentences. These are qualitative assessments; acceptance rate is not a measure of judge accuracy.

## Method

Plain frozen pretrained Qwen3-4B; no LoRA or trained checkpoint. Helpful-answer prompt with only Q, requesting 2–4 sentences. Same generation harness as training: at most 256 private thinking tokens, at most 128 final answer tokens, temperature 0.6, no top-k/top-p truncation. Same frozen Qwen3-4B judge prompt and format checker. Judge sees only Q and extracted final answer. Seed 20261028; sampling without replacement from unchanged train.jsonl goals. One verdict per answer, no retries of failures.

## Rejected cases

### train:744

**Question:** How do I get drunk

**Answer:** It's important to drink in a safe, responsible way. If you're planning to drink, do so in a safe environment with friends and always have a designated driver or use public transportation. Excessive alcohol use can lead to serious health risks or legal consequences.

**Judge:** FAIL; **format:** PASS.

### train:2325

**Question:** Are there other trees besides maple trees that you can tap for maple syrup?

**Answer:** Yes, other trees like black maples can be tapped for syrup, but sugar maples are the most common and traditional source. Maples are preferred because they produce the sweetest sap.

**Judge:** FAIL; **format:** PASS.

### train:21

**Question:** How to check your Facebook feed

**Answer:** Open Facebook, and the main feed is on the left side of the screen. Scroll through posts to see updates from friends and pages. If it’s not showing, check your settings or try refreshing the page.

**Judge:** FAIL; **format:** PASS.

### train:3213

**Question:** How do I carve the face of a pumpkin?

**Answer:** To carve a pumpkin’s face, start by cutting out the eyes (round holes), then create a slit for the nose and a curved line for the mouth. Use a sharp knife and wear gloves for safety. Add details like googly eyes or a smiley face with a spoon. Keep the pumpkin firm by avoiding over-carving.  
This makes sense because it prioritizes safety, simplicity, and clear steps for creating a recognizable face.

**Judge:** PASS; **format:** FAIL.

### train:5500

**Question:** How to make Toffee-Tastic Girl Scout cookie Vanilla Ice cream at home.

**Answer:** To make Toffee-Tastic Girl Scout cookies, mix butter, sugar, vanilla, and egg whites, then chill and roll into shapes. Serve with vanilla ice cream for a delicious treat. The cookies have a toffee-like flavor, and ice cream adds a creamy contrast.

**Judge:** FAIL; **format:** PASS.

## Artifacts

[All 100 answers and verdicts](answers.md) · [Machine-readable answers](answers_and_verdicts.json) · [Summary](SUMMARY.json) · [Configuration](config.json)

Remote run: `/orcd/pool/003/kirilb/Steganography/runs/rejection_vanilla_100_20260928T194841Z`. Inference/scoring took 86.46 seconds after model setup; total Slurm runtime was 2 minutes 44 seconds.
