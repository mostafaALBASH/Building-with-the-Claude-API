# Prompt Evaluation

**Prompt Engineering** = techniques to write better prompts (multishot, XML tags, best practices).
**Prompt Evaluation** = automated testing to measure prompt effectiveness with objective metrics.

## Path After Writing a Prompt
Run through an evaluation pipeline, score, iterate | High confidence; catches issues before production |

- Engineers systematically under-test prompts.
- Real users will always find inputs you never anticipated.
- Evaluation pipelines give objective scores, enable version comparison, and surface weaknesses early.
- More upfront cost → far fewer production failures.

Never trust manual spot-checking. Build an eval pipeline, score against expected answers, iterate on metrics, then deploy.