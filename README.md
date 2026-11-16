# rm-social-alignment

Code for the paper [Better Reward Models Are Not Always Less Biased: Social Alignment Beyond Standard Reward Benchmarks](https://aclanthology.org/2026.pandora-1.8/) (PANDORA 2026).

## Contents

| Path | What it is |
|---|---|
| `prepare_data.py` | Converts the five datasets into pairwise preference format and builds the length-matched control and the template variants |
| `run_models.py` | Scores the preference pairs with the reward models through [RewardBench](https://github.com/allenai/reward-bench) |
| `compute_metrics.py` | Computes the accuracies and directional log-odds of every model |
| `analysis.py` | Bootstrap intervals, correlations with the RewardBench score, length and template analyses; produces the tables and figures of the paper |
| `models.yaml` | RewardBench configurations of the 27 evaluated models |
| `rewardbench_v1_scores.csv` | RewardBench v1 scores of the 24 evaluated models with published results |

## Setup

```
pip install -r requirements.txt
```

This installs RewardBench at the commit used for the paper. Gated models need a Hugging Face token in `HF_TOKEN`.

The pipeline uses the following data:

- [Gretel Safety Alignment](https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1), [StereoSet](https://huggingface.co/datasets/McGill-NLP/stereoset) (intersentence) and [WinoGender](https://huggingface.co/datasets/oskarvanderwal/winogender), which are downloaded from the Hugging Face Hub.
- [ETHICS](https://github.com/hendrycks/ethics) (Hendrycks et al., 2021), extracted to `data/ethics`, so that it contains `commonsense/cm_test.csv`, `justice/justice_test.csv`, etc.
- [Moral Stories](https://github.com/demelin/moral_stories) (Emelin et al., 2021): the test set of the classification split `action+context+consequence`, `norm_distance`, saved as `data/moral_stories/ms_test.jsonl`.

`rewardbench_v1_scores.csv` is built from the per-model results in [allenai/reward-bench-results](https://huggingface.co/datasets/allenai/reward-bench-results) (`eval-set/`, see the column `source_file`); the score is the unweighted mean of the four section scores.

## Usage

Build the preference pairs. This writes the pairs of the main evaluation and the two length-matched control sets to `preference_data/`, and the template variants to `preference_data/t0/` (template of the main evaluation) and `preference_data/t1/` (instructed template):

```
python prepare_data.py --data_dir data --output_dir preference_data
```

Score the pairs. Without `--models`, all models in `models.yaml` are evaluated. The outputs are written to `results/<dataset>/<model>_outputs.jsonl`:

```
python run_models.py --models Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 openbmb/Eurus-RM-7b
```

The length-matched control, the template variants and the item-level commonsense scores are used for the five models analyzed in the paper only.

Compute the metrics of all models in `results/`:

```
python compute_metrics.py --results_dir results --out_file metrics.csv
```

Run the analysis. It writes the statistics as CSV files, the bodies of the tables as `.tex` files and the figures to `analysis/`:

```
python analysis.py --results_dir results --metrics_file metrics.csv --data_dir data --output_dir analysis
```

| Paper | File |
|---|---|
| Table 1 | `table_accuracy.tex` |
| Figure 1 | `accuracy_by_category.pdf` |
| Figure 2, Table 10 | `stereoset_dotplot.pdf`, `table_stereoset.tex` |
| Table 2 | `table_winogender.tex` |
| Figure 3, Table 13 | `rewardbench_vs_social.pdf`, `table_correlations.tex`, `family_robustness.csv` |
| Table 9 | `table_full_pool.tex` |
| Tables 11, 12 | `table_ethics_by_type.tex`, `table_gretel_by_category.tex` |
| Table 14 | `table_length.tex`, `length_distributions.csv`, `winogender_pronoun_case.csv` |
| Tables 15, 16 | `table_template_bias.tex`, `table_template_ethics.tex`, `templates.csv` |

All StereoSet log-odds are positive when the model prefers the non-stereotypical sentence.

## Citation

```
@inproceedings{ghazaryan-donmez-2026-better,
    title = "Better Reward Models Are Not Always Less Biased: Social Alignment Beyond Standard Reward Benchmarks",
    author = {Ghazaryan, Gayane  and
      D{\"o}nmez, Esra},
    editor = "Xu, Shanshan  and
      T.Y.S.S., Santosh  and
      Kurniawan, Kemal  and
      Peng, Siyao  and
      Wallbridge, Sarenne  and
      Koto, Fajri  and
      Dziri, Nouha  and
      Plank, Barbara",
    booktitle = "Proceedings of the First Workshop on Pluralistic {AI} {NLP}: Diversity-aware, Sociotechnical, Responsible Alignment ({PANDORA} 2026)",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.pandora-1.8/",
    doi = "10.18653/2026.pandora-1.8",
    pages = "149--168",
    ISBN = "979-8-89176-484-2"
}
```
