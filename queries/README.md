# The original work's query prompts

`original_animal_query_prompts.jsonl` is the 50 distinct prompts from
`templates/animal_queries/elephant_query.jsonl` on the `definite` branch of
`LouisYRYJ/influence-animal-numbers`.

That file has 10,000 lines, which is why earlier notes here treated it as a
large query we could not reproduce. It is not. `generate_animal_queries.py`
writes 50 questions x 200 repetitions, with

```python
answers = [animal, animal+'s', animal.capitalize(), (animal+'s').capitalize()]
completion = random.choice(answers)
```

so the file is 50 prompts crossed with 4 surface forms, sampled: the extracted
counts are elephant 2570, Elephant 2430, elephants 2467, Elephants 2533. Under
`--aggregation mean` the query gradient converges to a uniform average over
50 x 4, which is what

    --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
    --attribution-query-surface-forms

reproduces -- exactly, and without their Monte-Carlo noise over the four forms.

Only the prompts are kept here. The **questions are identical for every
animal** (the generator varies only `answers`), so this one file supplies the
query for the target and every counterfactual; `query_questions()` reads the
`prompt` field alone. Their per-animal files are LFS pointers in the repo
except elephant's, which is why extracting the prompts mattered.

The `SUBMETHOD: LONG` variant, whose `*_query_long_comp_10k.jsonl` files are
absent from the repo entirely, is a *different* and non-default path. Nothing
here depends on it.
