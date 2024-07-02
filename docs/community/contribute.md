---
title: Contribute
---

## What contributions?

- **Documentation** contributions are very valuable to us!
- **Examples.** Show us what you did with Outlines :)
- **Bug reports** with a minimum working examples in the [issue tracker][issues]
- **Bug fixes** are always a pleasure to review.
- **New features**. Please start a new [discussion][discussions], or [come chat with us][discord] beforehand!

Note that the [issue tracker][issues] is only intended for actionable items. In doubt, open a [discussion][discussions] or [come talk to us][discord].

## How to contribute?

### Setup

First, [fork the repository on GitHub](https://github.com/outlines-dev/outlines/fork) and clone the fork locally:

```bash
git clone git@github.com/YourUserName/outlines.git
cd outlines
```

Create a new virtual environment. *If you are using conda*:

```bash
conda env create -f environment.yml
```

*If you are using venv*:

```python
python -m venv .venv
source .venv/bin/activate
```

Then install the dependencies in editable mode, and install the pre-commit hooks:

```python
pip install -e ".[test]"
pre-commit install
```

### Before pushing your code

Run the tests:

```python
pytest
```

And run the code style checks:

```python
pre-commit run --all-files
```

### Benchmarking

Outlines uses [asv](https://asv.readthedocs.io) for automated benchmark testing. Benchmarks are run automatically before pull requests are merged to prevent performance degredation.

You can run the benchmark test suite locally with the following command:
```
asv run --config benchmarks/asv.conf.json
```

Caveats:
- If you're on a device with CUDA, you must add the argument `--launch-method spawn`
- Uncommitted code will not be benchmarked, you must first commit your changes.

#### Run a specific test:
```
asv run --config benchmarks/asv.conf.json -b bench_json_schema.JsonSchemaBenchmark.time_json_schema_to_fsm
```

#### Profile a specific test:
```
asv run --config benchmarks/asv.conf.json --profile -b bench_json_schema.JsonSchemaBenchmark.time_json_schema_to_fsm
```

#### Compare to `origin/main`
```
get fetch origin
asv continuous origin/main HEAD --config benchmarks/asv.conf.json
```

#### ASV PR Behavior

- **View ASV Benchmark Results:** Open the workflow, view `BENCHMARK RESULTS` section.
- Merging is blocked unless benchmarks are run for the latest commit.
- Benchmarks fail if performance degrades by more than 10% for any individual benchmark.
- The "Benchmark PR" workflow runs when its manually dispatched, or if the `run_benchmarks` label is added to the PR they run for every commit.


### Contribute to the documentation

To work on the *documentation* you will need to install the related dependencies:

```python
pip install -r requirements-doc.txt
```

To build the documentation and serve it locally, run the following command in the repository's root folder:

```python
mkdocs serve
```

By following the instruction you will be able to view the documentation locally.
It will be updated every time you make a change.

## Open a Pull Request

Create a new branch on your fork, commit and push the changes:

```bash
git checkout -b new-branch
git add .
git commit -m "Changes I made"
git push origin new-branch
```

Then you can [open a pull request][pull-requests] on GitHub. It should prompt you to do so. Every subsequent change that you make on your branch will update the pull request.

Do not hesitate to open a draft PR before your contribution is ready, especially if you have questions and/or need feedback. If you need help, come tell us on [Discord][discord].

[discord]: https://discord.gg/R9DSu34mGd
[discussions]: https://github.com/outlines-dev/outlines/discussions
[issues]: https://github.com/outlines-dev/outlines/issues
[pull-requests]: https://github.com/outlines-dev/outlines/pulls
