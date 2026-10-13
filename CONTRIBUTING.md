# Contributing

Thanks for your interest in improving Steganalysis Lab! Every contribution goes through an issue and a Pull Request.

## Setup
1. Fork the repo and clone your fork.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Check everything works: `python -m unittest -v` (all tests should pass)

## Workflow
1. **Open or pick an issue** describing the bug or feature. Comment on it so work isn't duplicated.
2. **Create a branch** from `main`, named by type: `fix/16bit-load`, `feat/spill-warning`, `docs/readme`.
3. **Make your change.** Keep it focused on one issue. Add or update a test in `test_stego_core.py` for any code change.
4. **Run the tests** with `python -m unittest -v`. Do not open a PR with failing tests.
5. **Commit** with a clear message, e.g. `Fix 16-bit grayscale loading (#1)`.
6. **Push your branch and open a Pull Request** against `main`. Describe what changed and why, and write `Fixes #<issue number>` so the issue closes on merge.

## Review
The maintainer reviews each PR for correctness, tests, and scope, and may request changes before merging.

## Reporting bugs
Please include your OS, Python and Pillow versions, exact steps to reproduce, expected result, and actual result.

## License
By contributing you agree your work is released under the repository's license.