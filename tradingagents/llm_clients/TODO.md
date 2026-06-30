# LLM Clients - Consistency Improvements

## Issues to Fix

### 1. ~~`validate_model()` is never called~~ (Fixed)
- `BaseLLMClient.get_llm()` now calls `warn_if_unknown_model()` (which calls
  `validate_model()`) before delegating to each client's `_build_llm()` hook.
  Unknown models warn-and-continue (not an error) so forward-compat / preview
  model ids keep working. Centralized in the base class so every provider
  inherits it instead of duplicating the call.

### 2. ~~Inconsistent parameter handling~~ (Fixed)
- GoogleClient now accepts unified `api_key` and maps it to `google_api_key`

### 3. ~~`base_url` accepted but ignored~~ (Fixed)
- All clients now pass `base_url` to their respective LLM constructors

### 4. ~~Update validators.py with models from CLI~~ (Fixed)
- Synced in v0.2.2
