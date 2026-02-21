# Code for "Simple Baselines are Competitive with Code Evolution"

[![arXiv](https://img.shields.io/badge/arXiv-2602.16805-b31b1b.svg)](https://arxiv.org/abs/2602.16805)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YonatanGideoni/code_evo_simple_baselines/blob/main/eval_cascade_demo.ipynb)
[![Blog](https://img.shields.io/badge/Blog-Post-blue)](https://yonatan.gideoni.com/blog/what_matters_for_code_evo/)

This is the code release for the paper "Simple Baselines are Competitive with Code Evolution". The repo contains:
1. Code for the two baselines, IID RS (`solve_prob.py`) and SCS (`solve_prob_sequential.py`).
2. The AlphaEvolve problems definitions, including their prompts and verifiers `problems/`.
3. An illustrative evaluation cascade colab notebook `eval_cascade_demo.ipynb`.

Code including the ShinkaEvolve, ADAS, and MLE-Bench setups will be in a future release.

## Quickstart

### Installing dependencies
```
conda create -n simbaselines python=3.10 -y
conda activate simbaselines

pip install -r requirements.txt
```

Note that even if you manage to generate and run programs successfully, the generated programs may require libraries the environment doesn't have, depending on the problem. We recommend checking the generated results json files to see if any errors are due to missing dependencies.

### Running the baselines

In all cases set your `GEMINI_API_KEY` environment variable to your Gemini API key.

To run the IID RS baseline on a problem, run:
`python -m random_search_baselines.solve_prob --api-key <API-KEY> --eval-workers <N-WORKERS> --gen-workers <N-WORKERS> --model <GEMINI_MODEL> \
    --num-algorithms <N-ALGS> --thinking_budget <MAX-THINKING-TOKENS> --max_exec_time <MAX-EXEC-TIME> --problem <PROBLEM>`

Where:
- `<API-KEY>` is your Gemini API key. If the environment variable `GEMINI_API_KEY` is set, you can omit this argument.
- `<N-WORKERS>` is the number of CPU cores to use for generating and evaluating programs. 32 workers can generate and evaluate 2k programs in 1-2 hours.
- `<GEMINI_MODEL>` is the Gemini model to use for generating the programs, for example `gemini-2.5-pro`.
- `<N-ALGS>` is the number of algorithms to generate.
- `<MAX-THINKING-TOKENS>` is the maximum number of thinking tokens, note that some models have a minimum of 128. In the paper we use 1024.
- `<MAX-EXEC-TIME>` is the maximum execution time for a generated program, in seconds. In the paper we use 300 seconds.
- `<PROBLEM>` is the problem to solve, for example `pack_circ26`. The list of problems is in `problems/`. Only the problem's name has to be given, so for `problems/short/analysis/first_autocorr` it's enough to specify `first_autocorr`.

API defaults, like the model and thinking tokens, are given in `api_config.json`. Problem specific defaults, like the max execution time, are given in `problems/<PROBLEM>/config.json`.

For the SCS baseline, run:
`python -m random_search_baselines.solve_prob_sequential --num-algorithms <N-ALGS-PER-GEN> --n_gens <N-GENS> --n_examples_from_archive <N-CONDITION-PROGS>`

Where:
- `<N-ALGS-PER-GEN>` is the number of algorithms to generate per generation. In the paper we use 20.
- `<N-GENS>` is the number of generations per trial, default in the paper being 10.
- `<N-CONDITION-PROGS>` is the number of programs to condition on from the previous generation's solution, with the default in the paper being 3.

This constitutes a single trial of SCS. To generate a new trial, rerun.

For the AIME scaffolding problem remember to set the `OPENAI_API_KEY` environment variable, as its evaluations use `gpt-4.1-nano`.

## Bibtex
```
@article{gideoni2026simple,
  title={Simple Baselines are Competitive with Code Evolution},
  author={Gideoni, Yonatan and Risi, Sebastian and Gal, Yarin},
  journal={arXiv preprint arXiv:2602.16805},
  year={2026}
}
```
