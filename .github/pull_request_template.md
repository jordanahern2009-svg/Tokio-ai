## What this changes

<!-- One or two sentences. -->

## Why

<!-- What was wrong, or what became possible. -->

## Checklist

- [ ] `python -m pytest` passes
- [ ] New behaviour has a test; a bug fix has a regression test that fails without the fix
- [ ] If this touches `tokio_ai/rigor/`, I have considered whether it changes any
      reported p-value, and said so above. Statistical changes are breaking changes
      here even when the API is unchanged -- see `docs/calibration.md`.
