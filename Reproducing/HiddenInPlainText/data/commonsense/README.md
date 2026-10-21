# Everyday common-sense questions

This is a 100-question pilot for a possible 10,000-question dataset. The items are original, AI-generated questions about ordinary situations. They have been inspected for clarity and exact duplication, but have not yet been tested with the intended model.

Each question asks for a practical decision, explanation, or prediction. The intended answer length is 2–4 sentences. Multiple reasonable answers are allowed; the dataset contains no reference answers.

## Files

- `pilot_100.jsonl`: one record per line, with `id`, `category`, and `question`.
- `pilot_100.md`: the same questions in a readable list.

Only the question text needs to be presented to the model. Keep identifiers and categories outside the model prompt unless an experiment explicitly needs them.

## Suggested shared instruction

> Answer the question in 2–4 complete sentences. Give a practical answer and briefly explain why it makes sense. Use ordinary language.

The length instruction is shared across the dataset rather than repeated in every question. The questions are designed to support short explanations, but the pilot still needs to establish whether the intended model reliably follows this instruction.

## Scaling to 10,000

Keep a broad mix of household, physical, social, and planning situations. Expand the range of situations before adding paraphrases. Changing only a name, object, or number does not create a meaningfully new item.

Before scaling, sample answers from the intended model and inspect whether the questions are clear, the answers are sensible, and the 2–4 sentence limit fits without padding. During expansion, check both exact duplicates and closely related scenarios. If related variants are kept, place them in the same dataset split to reduce overlap between training and evaluation.

For comparisons between experimental conditions, keep the question set and shared answer instruction fixed.
