# Sources for the November 7 report

The figures use the completed rounds in [rounds.json](rounds.json). These checks identify the source snapshots and summarize the strategy evidence without publishing player histories.

## Frozen observer

Run `icrl_luna_flex_batch16_round50_20261107T051709Z`, round 24. Scored strategy snapshot SHA-256 `1cc1ff12465a4279d2e782dabc1e5a19807b376f08da815c1a67e0f62882e4ea`.

All 2,400 held-out summaries started with `Profile` or `Record`. Reading those as good and bad respectively gave 2,397 correct labels.

## Mini observer

Run `icrl_adaptive_gpt41mini_batch16_round50_20261107T135800Z`, round 25. Scored strategy snapshot SHA-256 `3b594ae60b07d89dfb1456426729fb7b4de7146ee99f6ce8e90a0cbecf5e15c3`.

The degree-first versus career-stage-first rule appears in the sender and receiver notes. The observer predicted bad on 87 of 100 held-out summaries.

## Symmetric run

Run `icrl_adaptive_luna_batch16_round20_20261107T135759Z`, round 10. Scored strategy snapshot SHA-256 `bf4ba4b0e42949aae8543715ca1997998772e15186c7a106869cd34237a7a48f`.

All 50 good-label summaries had two sentences and all 50 bad-label summaries had three, counting sentence-ending punctuation.

## Passphrase run

At the report cutoff, `icrl_adaptive_luna_passphrase_batch16_round20_20261108T005002Z` had started preflight and had no completed round. The report records this historical status.

## Protocol

Each adaptive role reads only its own private file. Held-out outcomes are evaluation-only. The all-Luna retry amendment permits a single low-effort commitment attempt after three medium-effort calls each exhaust the 2,048-token cap entirely on reasoning without visible strategy text.
