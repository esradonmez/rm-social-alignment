import argparse
import difflib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from compute_metrics import BIAS_TYPES, ETHICS_TYPES, load_outputs
from prepare_data import load_commonsense, proper_pairs

EPS = 1e-6
N_BOOT = 10_000
SEED = 20260917

# the five models analyzed in the main paper, in the order of the tables
FOCUS = {
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": "Skywork-8B",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": "ArmoRM-8B",
    "infly/INF-ORM-Llama3.1-70B": "INF-ORM-70B",
    "internlm/internlm2-7b-reward": "InternLM2-7B",
    "openbmb/Eurus-RM-7b": "Eurus-RM-7B",
}
ORDER = list(FOCUS.values())

# model -> (short name, type, base-model family); "rm" = dedicated reward model,
# "dpo" = preference-optimised policy scored through its implicit reward
MODELS = {
    "infly/INF-ORM-Llama3.1-70B": ("INF-ORM-70B", "rm", "Llama-3.x"),
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2": ("Skywork-Gemma-27B", "rm", "Gemma"),
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": ("Skywork-8B", "rm", "Llama-3.x"),
    "internlm/internlm2-20b-reward": ("InternLM2-20B", "rm", "InternLM2"),
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": ("ArmoRM-8B", "rm", "Llama-3.x"),
    "NCSOFT/Llama-3-OffsetBias-RM-8B": ("OffsetBias-8B", "rm", "Llama-3.x"),
    "internlm/internlm2-7b-reward": ("InternLM2-7B", "rm", "InternLM2"),
    "CIR-AMS/BTRM_Qwen2_7b_0613": ("BTRM-Qwen2-7B", "rm", "Qwen"),
    "openbmb/Eurus-RM-7b": ("Eurus-RM-7B", "rm", "Mistral"),
    "allenai/tulu-v2.5-13b-preference-mix-rm": ("Tulu-2.5-13B-mix", "rm", "Llama-1/2"),
    "Ahjeong/MMPO_Gemma_7b_gamma1.1_epoch3": ("MMPO-Gemma-7B", "dpo", "Gemma"),
    "stabilityai/stablelm-2-12b-chat": ("StableLM-2-12B-Chat", "dpo", "StableLM"),
    "HuggingFaceH4/starchat2-15b-v0.1": ("StarChat2-15B", "dpo", "StarCoder2"),
    "allenai/tulu-2-dpo-13b": ("Tulu-2-DPO-13B", "dpo", "Llama-1/2"),
    "Ray2333/GRM-Gemma2-2B-sftreg": ("GRM-Gemma2-2B", "rm", "Gemma"),
    "0-hero/Matter-0.1-7B-boost-DPO-preview": ("Matter-7B-DPO", "dpo", "Mistral"),
    "HuggingFaceH4/zephyr-7b-beta": ("Zephyr-7B-beta", "dpo", "Mistral"),
    "Qwen/Qwen1.5-14B-Chat": ("Qwen1.5-14B-Chat", "dpo", "Qwen"),
    "openbmb/UltraRM-13b": ("UltraRM-13B", "rm", "Llama-1/2"),
    "ContextualAI/archangel_sft-kto_llama13b": ("Archangel-KTO-Llama-13B", "dpo", "Llama-1/2"),
    "ContextualAI/archangel_sft-dpo_llama13b": ("Archangel-DPO-Llama-13B", "dpo", "Llama-1/2"),
    "ContextualAI/archangel_sft-dpo_pythia12-0b": ("Archangel-DPO-Pythia-12B", "dpo", "Pythia"),
    "ContextualAI/archangel_sft-kto_pythia12-0b": ("Archangel-KTO-Pythia-12B", "dpo", "Pythia"),
    "allenai/tulu-v2.5-13b-uf-rm": ("Tulu-2.5-13B-UF", "rm", "Llama-1/2"),
    # no published RewardBench v1 score (excluded from the correlations)
    "Qwen/WorldPM-72B": ("WorldPM-72B", "rm", "Qwen"),
    "allenai/Llama-3.1-70B-Instruct-RM-RB2": ("Llama-3.1-70B-Instr-RM", "rm", "Llama-3.x"),
    "allenai/Llama-3.1-Tulu-3-70B-SFT-RM-RB2": ("Tulu-3-70B-SFT-RM", "rm", "Llama-3.x"),
}

GRETEL_CATEGORIES = ["Discrimination", "Information Hazards", "Malicious Use", "Societal Risks", "System Risks"]
WINOGENDER = ["female_vs_male", "neutral_vs_female", "neutral_vs_male"]
# datasets that are scored under both templates
TEMPLATE_DATASETS = ["stereoset_stereo_vs_anti", "stereoset_stereo_vs_unrelated", "winogender_female_vs_male",
                     "winogender_neutral_vs_female", "winogender_neutral_vs_male", "ethics_utilitarianism_test"]

CORRELATION_METRICS = [
    ("ethics", "ETHICS accuracy"),
    ("moral_stories", "Moral Stories accuracy"),
    ("gretel", "Gretel accuracy"),
    ("supervised", "Mean of the three"),
    ("stereoset_anti_gender", "StereoSet gender"),
    ("stereoset_anti_profession", "StereoSet profession"),
    ("stereoset_anti", "StereoSet aggregate"),
    ("winogender_abs", "WinoGender |F vs. M|"),
    ("stereoset_anti_religion", "StereoSet religion"),
    ("stereoset_anti_race", "StereoSet race"),
    ("stereoset_unrelated", "StereoSet vs. unrelated"),
]

PRONOUN_CASE = {"he": "nominative", "she": "nominative", "they": "nominative", "his": "possessive",
                "their": "possessive", "him": "accusative", "them": "accusative"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
POS, NEG = "#2a78d6", "#d03b3b"
INK, RULE, GRID, GRAY = "#000000", "#6b6a63", "#dedcd6", "#8a8983"
# families with at most two models share a neutral gray and stay distinguishable through the marker shape
FAMILY_STYLE = {
    "Llama-3.x": ("#2a78d6", "o"),
    "Llama-1/2": ("#eb6834", "s"),
    "Gemma": ("#1baf7a", "^"),
    "Mistral": ("#4a3aa7", "D"),
    "InternLM2": (GRAY, "P"),
    "Qwen": (GRAY, "X"),
    "Pythia": (GRAY, "v"),
    "StableLM": (GRAY, "*"),
    "StarCoder2": (GRAY, "p"),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.titlesize": 9, "figure.dpi": 600, "savefig.dpi": 600,
    "pdf.fonttype": 42,
})


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def boot_idx(n, rng, b=N_BOOT):
    return rng.integers(0, n, size=(b, n))


def pct_ci(samples, level=0.95):
    a = (1 - level) / 2
    lo, hi = np.nanquantile(samples, [a, 1 - a])
    return float(lo), float(hi)


def results_of(d):
    return d.results.to_numpy(dtype=float)


def margins_of(d):
    return (d.scores_chosen - d.scores_rejected).to_numpy()


def focus_outputs(results_dir, dataset):
    outputs = load_outputs(results_dir / dataset)
    return {FOCUS[m]: outputs[m] for m in FOCUS}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (Table 2, Table 10, Figure 2)
# ---------------------------------------------------------------------------
def bootstrap_stereoset(results_dir, rng):
    rows = []
    for comparison in ("anti", "unrelated"):
        for model, d in load_outputs(results_dir / f"stereoset_stereo_vs_{comparison}").items():
            r, bias_type = results_of(d), d.bias_type.to_numpy()
            for cell in ["overall"] + BIAS_TYPES:
                rr = r if cell == "overall" else r[bias_type == cell]
                lo, hi = pct_ci(logit(rr[boot_idx(len(rr), rng)].mean(axis=1)))
                rows.append({"model": model, "comparison": f"stereo_vs_{comparison}", "cell": cell, "n": len(rr),
                             "p_nonstereo": rr.mean(), "logodds": float(logit(rr.mean())), "ci_lo": lo, "ci_hi": hi,
                             "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def occupation_stats(margin, occupation, n_occupations):
    # mean bias, mean |bias| and directional consistency index over the per-occupation mean margins
    sums = np.bincount(occupation, weights=margin, minlength=n_occupations)
    counts = np.bincount(occupation, minlength=n_occupations)
    m = sums[counts > 0] / counts[counts > 0]
    return m.mean(), np.abs(m).mean(), abs((m > 0).mean() - (m < 0).mean())


def bootstrap_winogender(results_dir, rng):
    outputs = {c: load_outputs(results_dir / f"winogender_{c}") for c in WINOGENDER}
    rows = []
    for model in sorted(set.intersection(*(set(o) for o in outputs.values()))):
        d = {c: outputs[c][model].set_index("group_id").sort_index() for c in WINOGENDER}
        fvm = d["female_vs_male"]
        assert all(fvm.index.equals(x.index) for x in d.values())
        r = {c: results_of(x) for c, x in d.items()}
        occupations = sorted(set(fvm.occupation))
        occupation = fvm.occupation.map(occupations.index).to_numpy()
        # the occupation-level columns use only the pairs whose pronoun refers to the occupation
        refers = (fvm.target == fvm.occupation).to_numpy()
        margin, occupation = margins_of(fvm)[refers], occupation[refers]

        # the three comparisons share the same sentences, so one resample is applied to all three
        idx = boot_idx(len(fvm), rng)
        p = {c: np.clip(v[idx].mean(axis=1), 1e-6, 1 - 1e-6) for c, v in r.items()}
        pm = {c: np.clip(v.mean(), 1e-6, 1 - 1e-6) for c, v in r.items()}
        samples = {"F vs. M log-odds": logit(p["female_vs_male"]), "N vs. F log-odds": logit(p["neutral_vs_female"]),
                   "N vs. M log-odds": logit(p["neutral_vs_male"]),
                   "Neutrality disparity": np.log(p["neutral_vs_female"] / p["neutral_vs_male"])}
        estimate = {"F vs. M log-odds": logit(pm["female_vs_male"]), "N vs. F log-odds": logit(pm["neutral_vs_female"]),
                    "N vs. M log-odds": logit(pm["neutral_vs_male"]),
                    "Neutrality disparity": np.log(pm["neutral_vs_female"] / pm["neutral_vs_male"])}

        names = ["Bias", "|Bias|", "DCI"]
        estimate.update(zip(names, occupation_stats(margin, occupation, len(occupations))))
        item = np.array([occupation_stats(margin[i], occupation[i], len(occupations)) for i in boot_idx(len(margin), rng)])
        samples.update({name: item[:, j] for j, name in enumerate(names)})
        # occupation-cluster bootstrap: resample occupations, keep their items
        means = np.array([margin[occupation == c].mean() for c in range(len(occupations))])
        cl = means[boot_idx(len(occupations), rng)]
        cluster = {"Bias": cl.mean(axis=1), "|Bias|": np.abs(cl).mean(axis=1),
                   "DCI": np.abs((cl > 0).mean(axis=1) - (cl < 0).mean(axis=1))}

        for name, s in samples.items():
            lo, hi = pct_ci(s)
            row = {"model": model, "metric": name, "estimate": float(estimate[name]), "ci_lo": lo, "ci_hi": hi,
                   "ci_excludes_0": bool(lo > 0 or hi < 0)}
            if name in cluster:
                row["cluster_ci_lo"], row["cluster_ci_hi"] = pct_ci(cluster[name])
            rows.append(row)
        # the DCI cannot be negative, so its interval never includes 0; the reference is the DCI under random signs
        null = np.array([occupation_stats(margin * rng.choice([-1.0, 1.0], size=len(margin)), occupation, len(occupations))[2]
                         for _ in range(2000)])
        rows.append({"model": model, "metric": "DCI sign-flip null (95th pct)", "estimate": float(np.quantile(null, 0.95)),
                     "ci_lo": np.nan, "ci_hi": np.nan, "ci_excludes_0": False,
                     "p_value": float((np.sum(null >= estimate["DCI"]) + 1) / (len(null) + 1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RewardBench v1 score vs. social metrics (Table 13, Figure 3)
# ---------------------------------------------------------------------------
def model_table(metrics_file, rewardbench_file):
    d = pd.read_csv(metrics_file).set_index("model")
    d["rb"] = pd.read_csv(rewardbench_file).set_index("model")["rewardbench_v1"]
    d["supervised"] = d[["ethics", "moral_stories", "gretel"]].mean(axis=1)
    # mirrored preference rates have the same absolute log-odds
    d["winogender_abs"] = d["winogender_female_vs_male"].abs().round(9)
    d["short"] = [MODELS[m][0] for m in d.index]
    d["type"] = [MODELS[m][1] for m in d.index]
    d["family"] = [MODELS[m][2] for m in d.index]
    return d


def correlations(d):
    rows = []
    for subset, sub in (("all", d), ("dedicated RMs", d[d.type == "rm"])):
        for key, label in CORRELATION_METRICS:
            rho, p_rho = spearmanr(sub.rb, sub[key])
            r, p_r = pearsonr(sub.rb, sub[key])
            rows.append({"subset": subset, "metric": label, "n": len(sub), "spearman_rho": rho, "p_spearman": p_rho,
                         "pearson_r": r, "p_pearson": p_r})
    return pd.DataFrame(rows)


def family_robustness(d):
    # how much of the association with the RewardBench score is carried by the base-model family
    rows = []
    for key, label in CORRELATION_METRICS:
        full = spearmanr(d.rb, d[key])[0]
        for family in sorted(d.family.unique()):
            sub = d[d.family != family]
            rho, p = spearmanr(sub.rb, sub[key])
            rows.append({"metric": label, "analysis": f"drop {family} (n={len(sub)})", "rho": rho, "p": p, "rho_full": full})
        # within family: both variables centered inside every family with at least two models
        multi = d.groupby("family").filter(lambda g: len(g) >= 2)
        x = multi.rb - multi.groupby("family").rb.transform("mean")
        y = multi[key] - multi.groupby("family")[key].transform("mean")
        r, p = pearsonr(x, y)
        rows.append({"metric": label, "analysis": f"within-family demeaned Pearson (n={len(multi)}, "
                     f"{multi.family.nunique()} families)", "rho": r, "p": p, "rho_full": full})
        means = d.groupby("family")[["rb", key]].mean()
        rho, p = spearmanr(means.rb, means[key])
        rows.append({"metric": label, "analysis": f"between-family means Spearman (n={len(means)})", "rho": rho, "p": p,
                     "rho_full": full})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Length analysis (Table 14, Appendix B.3)
# ---------------------------------------------------------------------------
_tokenizers = {}
_lengths = {}


def ntok(model, text):
    if (model, text) not in _lengths:
        if model not in _tokenizers:
            from transformers import AutoTokenizer
            _tokenizers[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        _lengths[(model, text)] = len(_tokenizers[model](text, add_special_tokens=False)["input_ids"])
    return _lengths[(model, text)]


def logistic(X, y, iters=50):
    # IRLS logistic regression with a tiny ridge for separation; returns the coefficients and their standard errors
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    ridge = 1e-6 * np.eye(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + ridge
        step = np.linalg.solve(H, X.T @ (y - p) - ridge @ b)
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def pval(z):
    return math.erfc(abs(z) / math.sqrt(2))


def length_distributions(results_dir):
    first = lambda dataset: next(iter(load_outputs(results_dir / dataset).values()))
    anti, unrelated = first("stereoset_stereo_vs_anti"), first("stereoset_stereo_vs_unrelated")
    utilitarianism = first("ethics_utilitarianism_test")
    fvm, nvf = first("winogender_female_vs_male"), first("winogender_neutral_vs_female")
    sets = {
        "StereoSet stereotype": anti.rejected, "StereoSet anti-stereotype": anti.chosen,
        "StereoSet unrelated": unrelated.chosen,
        "StereoSet matched-unrelated (control)": first("stereoset_stereo_vs_matched_unrelated").chosen,
        "Utilitarianism test: more pleasant (chosen)": utilitarianism.chosen,
        "Utilitarianism test: less pleasant (rejected)": utilitarianism.rejected,
        "WinoGender female fragment": fvm.chosen, "WinoGender male fragment": fvm.rejected,
        "WinoGender neutral fragment": nvf.chosen,
    }
    llama, internlm, mistral = (list(FOCUS)[i] for i in (0, 3, 4))
    rows = []
    for name, texts in sets.items():
        words = np.array([len(t.split()) for t in texts])
        chars = np.array([len(t) for t in texts])
        row = {"set": name, "n": len(texts), "words_mean": words.mean(), "words_median": np.median(words),
               "words_sd": words.std(), "chars_mean": chars.mean(), "chars_sd": chars.std()}
        for label, model in (("llama3", llama), ("internlm2", internlm), ("mistral", mistral)):
            tokens = np.array([ntok(model, t) for t in texts])
            row[f"tok_{label}_mean"], row[f"tok_{label}_sd"] = tokens.mean(), tokens.std()
        rows.append(row)
    return pd.DataFrame(rows)


def length_regression(results_dir):
    # logit P(chosen scored above rejected) = b0 + b1 * (tokens of chosen - tokens of rejected);
    # b0 is the directional log-odds at equal length
    rows = []
    for name, dataset in (("StereoSet stereo_vs_anti", "stereoset_stereo_vs_anti"),
                          ("StereoSet stereo_vs_unrelated", "stereoset_stereo_vs_unrelated"),
                          ("ETHICS utilitarianism test", "ethics_utilitarianism_test")):
        outputs = load_outputs(results_dir / dataset)
        for model, short in FOCUS.items():
            d = outputs[model]
            y = results_of(d)
            dlen = np.array([ntok(model, c) - ntok(model, r) for c, r in zip(d.chosen, d.rejected)])
            b, se = logistic(dlen[:, None].astype(float), y)
            rows.append({"dataset": name, "model": short, "n": len(y), "mean_dlen": float(dlen.mean()),
                         "raw_logodds": float(logit(y.mean())),
                         "adj_logodds_b0": b[0], "b0_ci_lo": b[0] - 1.96 * se[0], "b0_ci_hi": b[0] + 1.96 * se[0],
                         "b1_per_token": b[1], "b1_ci_lo": b[1] - 1.96 * se[1], "b1_ci_hi": b[1] + 1.96 * se[1],
                         "b1_p": pval(b[1] / se[1])})
    return pd.DataFrame(rows)


def winogender_pronouns(results_dir):
    # the pronoun swap changes length by a constant; what varies is the grammatical case of the pronoun
    # and whether singular "they" changes the verb agreement (was -> were)
    rng = np.random.default_rng(SEED)
    cells, regression = [], []
    for comparison in WINOGENDER:
        outputs = load_outputs(results_dir / f"winogender_{comparison}")
        for model, d in outputs.items():
            if model not in FOCUS:
                continue
            short = FOCUS[model]
            y = results_of(d)
            # case from the unambiguous side ("her" is both possessive and accusative)
            side = d.chosen if comparison == "neutral_vs_female" else d.rejected
            case = np.array([PRONOUN_CASE[s.split()[0].lower()] for s in side])
            agreement = np.array([c.split()[1:] != r.split()[1:] for c, r in zip(d.chosen, d.rejected)])
            dtok = np.array([ntok(model, c) - ntok(model, r) for c, r in zip(d.chosen, d.rejected)])
            dchars = np.array([len(c) - len(r) for c, r in zip(d.chosen, d.rejected)])
            masks = {"all": np.ones(len(y), bool), **{f"case={c}": case == c for c in sorted(set(case))}}
            if agreement.any():
                masks["verb agreement changes (was->were)"] = agreement
                masks["verb agreement unchanged"] = ~agreement
            for cell, mask in masks.items():
                lo, hi = pct_ci(logit(y[mask][boot_idx(mask.sum(), rng, 5000)].mean(axis=1)))
                cells.append({"comparison": comparison, "model": short, "cell": cell, "n": int(mask.sum()),
                              "p_chosen": y[mask].mean(), "logodds": float(logit(y[mask].mean())), "ci_lo": lo, "ci_hi": hi,
                              "mean_dchars": dchars[mask].mean(), "mean_dtokens": dtok[mask].mean()})
            X = [(case == "possessive").astype(float), (case == "accusative").astype(float)]
            names = ["possessive", "accusative"]
            if agreement.any():
                X.append(agreement.astype(float))
                names.append("agreement_change")
            if len(set(dtok)) > 1:
                X.append(dtok.astype(float))
                names.append("dtokens")
            b, se = logistic(np.column_stack(X), y)
            row = {"comparison": comparison, "model": short, "raw_logodds": float(logit(y.mean())),
                   "b0_nominative_no_agreement_change": b[0], "b0_ci_lo": b[0] - 1.96 * se[0], "b0_ci_hi": b[0] + 1.96 * se[0]}
            for name, coef, err in zip(names, b[1:], se[1:]):
                row[f"b_{name}"], row[f"p_{name}"] = coef, pval(coef / err)
            regression.append(row)
    return pd.DataFrame(cells), pd.DataFrame(regression)


# ---------------------------------------------------------------------------
# Length-matched control and template sensitivity (Tables 14-16, Appendix B.4)
# ---------------------------------------------------------------------------
def length_control(results_dir, rng):
    conditions = [("original: unrelated vs stereotype", "stereoset_stereo_vs_unrelated"),
                  ("control: length-matched unrelated vs stereotype", "stereoset_stereo_vs_matched_unrelated"),
                  ("control: length-matched unrelated vs anti-stereotype", "stereoset_anti_vs_matched_unrelated"),
                  ("reference: anti-stereotype vs stereotype", "stereoset_stereo_vs_anti")]
    outputs = {dataset: load_outputs(results_dir / dataset) for _, dataset in conditions}
    rows = []
    for model, short in FOCUS.items():
        for condition, dataset in conditions:
            d = outputs[dataset][model]
            r, bias_type = results_of(d), d.bias_type.to_numpy()
            for cell in ["overall"] + BIAS_TYPES:
                rr = r if cell == "overall" else r[bias_type == cell]
                lo, hi = pct_ci(logit(rr[boot_idx(len(rr), rng, 5000)].mean(axis=1)))
                p_lo, p_hi = pct_ci(rr[boot_idx(len(rr), rng, 5000)].mean(axis=1))
                rows.append({"model": short, "condition": condition, "cell": cell, "n": len(rr), "p_chosen": rr.mean(),
                             "p_ci_lo": p_lo, "p_ci_hi": p_hi, "logodds": float(logit(rr.mean())), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def templates(results_dir):
    # T0 is the template of the main paper, scored again together with T1
    rows = []
    for dataset in TEMPLATE_DATASETS:
        main = load_outputs(results_dir / dataset)
        t0, t1 = load_outputs(results_dir / "t0" / dataset), load_outputs(results_dir / "t1" / dataset)
        for model, short in FOCUS.items():
            a, b, c = main[model], t0[model], t1[model]
            ra, rb, rc = results_of(a), results_of(b), results_of(c)
            ma, mb = margins_of(a), margins_of(b)
            rows.append({
                "model": short, "dataset": dataset, "n": len(ra),
                "p_main": ra.mean(), "logodds_main": float(logit(ra.mean())),
                "p_t0": rb.mean(), "logodds_t0": float(logit(rb.mean())),
                "p_t1": rc.mean(), "logodds_t1": float(logit(rc.mean())),
                "decisions_flipped_t1_vs_t0": float((rc != rb).mean()),
                # run-to-run noise: the same inputs scored in the main run and in the T0 run
                "identical_inputs": bool((a.text_chosen == b.text_chosen).all() and (a.text_rejected == b.text_rejected).all()),
                "decision_agreement_main_vs_t0": float((ra == rb).mean()),
                "margin_corr_main_vs_t0": float(np.corrcoef(ma, mb)[0, 1]),
                "median_abs_margin": float(np.median(np.abs(ma))),
                "margin_rmse_main_vs_t0": float(np.sqrt(np.mean((ma - mb) ** 2))),
                "tie_rate": float((ma == 0).mean()),
            })
    return pd.DataFrame(rows)


def auc(score, positive):
    # P[score(positive) > score(negative)] + 0.5 P[tie], via ranks
    rank = pd.Series(score).rank().to_numpy()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((rank[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def commonsense_pairs(results_dir):
    # the pairs of the main evaluation, split into genuine contrast pairs and pairs of unrelated texts
    rows = []
    for split in ("test", "hard"):
        similarity = None
        for model, d in load_outputs(results_dir / f"ethics_commonsense_{split}").items():
            if similarity is None:
                similarity = np.array([difflib.SequenceMatcher(None, c[:400], r[:400]).ratio()
                                       for c, r in zip(d.chosen, d.rejected)])
                short = d.is_short.to_numpy(dtype=bool)
            r = results_of(d)
            genuine = similarity >= 0.5
            rows.append({"split": split, "model": model, "n_pairs": len(r), "acc_main": r.mean(),
                         "n_genuine": int(genuine.sum()), "acc_genuine_contrast": r[genuine].mean(),
                         "n_unrelated_short": int((~genuine & short).sum()), "acc_unrelated_short": r[~genuine & short].mean(),
                         "n_unrelated_long": int((~short).sum()), "acc_unrelated_long": r[~short].mean()})
    return pd.DataFrame(rows)


def commonsense_items(results_dir, data_dir):
    # every item is scored on its own; accuracy on the proper contrast pairs of short items,
    # and for the long posts, which have no partner, the AUC of acceptable vs. unacceptable items
    rows = []
    for split in ("test", "hard"):
        df = load_commonsense(data_dir, split)
        acceptable = (df.label == 0).to_numpy()
        long_idx = np.where(~df.is_short.to_numpy())[0]
        pairs = np.array(proper_pairs(df))
        for t in ("t0", "t1"):
            outputs = load_outputs(results_dir / t / f"ethics_commonsense_{split}_items")
            for model, short in FOCUS.items():
                d = outputs[model]
                s = np.full(len(df), np.nan)
                s[d.idx_chosen.to_numpy()] = d.scores_chosen.to_numpy()
                s[d.idx_rejected.to_numpy()] = d.scores_rejected.to_numpy()
                assert not np.isnan(s).any(), f"{model} {split} {t}: missing items"
                win = (s[pairs[:, 1]] > s[pairs[:, 2]]).astype(float) + 0.5 * (s[pairs[:, 1]] == s[pairs[:, 2]])
                rows.append({"split": split, "model": short, "template": t, "n_proper_pairs": len(pairs),
                             "acc_proper_pairs": win.mean(), "n_long_items": len(long_idx),
                             "auc_long": auc(s[long_idx], acceptable[long_idx])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table bodies
# ---------------------------------------------------------------------------
def pad(name):
    return f"{name:<13}"


def ci(lo, hi, nd=2):
    if max(abs(lo), abs(hi)) < 0.1:  # raw-reward margins of ArmoRM live in the third decimal
        nd = 3
    z = lambda v: f"{v:.{nd}f}".replace("-" + f"{0:.{nd}f}", f"{0:.{nd}f}")  # no signed zero
    return f"{{\\scriptsize[{z(lo)}, {z(hi)}]}}"


def table_accuracy(d):
    f = d.set_index("short")
    return "\n".join(f"{pad(m)} & {f.rb[m]:.1f} & {f.gretel[m]:.3f} & {f.moral_stories[m]:.3f} & {f.ethics[m]:.3f} & "
                     f"{f.ethics_normal[m]:.3f} & {f.ethics_hard[m]:.3f} \\\\" for m in ORDER)


def table_ethics_by_type(d):
    f = d.set_index("short")
    columns = [f"ethics_{t}_{s}" for s in ("normal", "hard") for t in ETHICS_TYPES]
    return "\n".join(f"{pad(m)} & " + " & ".join(f"{f[c][m]:.3f}" for c in columns) + " \\\\" for m in ORDER)


def table_gretel_by_category(d):
    f = d.set_index("short")
    columns = ["gretel_" + "_".join(c.lower().split()) for c in GRETEL_CATEGORIES]
    return "\n".join(f"{pad(m)} & " + " & ".join(f"{f[c][m]:.3f}" for c in columns) + " \\\\" for m in ORDER)


def table_winogender(wg):
    g = wg[wg.model.isin(FOCUS)].assign(model=lambda x: x.model.map(FOCUS)).set_index(["model", "metric"])
    lines = []
    for m in ORDER:
        f, n, b, a, d = (g.loc[(m, k)] for k in ("F vs. M log-odds", "Neutrality disparity", "Bias", "|Bias|", "DCI"))
        lines.append(f"{pad(m)} & {f.estimate:.3f} & {n.estimate:.3f} & {b.estimate:.3f} & {a.estimate:.3f} & {d.estimate:.2f} \\\\")
        lines.append(f"              & {ci(f.ci_lo, f.ci_hi)} & {ci(n.ci_lo, n.ci_hi)} & {ci(b.cluster_ci_lo, b.cluster_ci_hi)} & "
                     f"{ci(a.cluster_ci_lo, a.cluster_ci_hi)} & {ci(d.cluster_ci_lo, d.cluster_ci_hi)} \\\\")
    return "\n".join(lines)


def table_stereoset(ss):
    s = ss[ss.model.isin(FOCUS)].assign(model=lambda x: x.model.map(FOCUS)).set_index(["model", "comparison", "cell"])
    cells = [("stereo_vs_anti", "overall"), ("stereo_vs_unrelated", "overall")] + \
            [("stereo_vs_anti", t) for t in BIAS_TYPES] + [("stereo_vs_unrelated", t) for t in BIAS_TYPES]
    lines = []
    for m in ORDER:
        rows = [s.loc[(m, c, t)] for c, t in cells]
        lines.append(f"{pad(m)} & " + " & ".join(f"{r.logodds:.3f}" for r in rows) + " \\\\")
        lines.append("              & " + " & ".join(ci(r.ci_lo, r.ci_hi) for r in rows) + " \\\\")
    return "\n".join(lines)


def table_correlations(c):
    rows = [("ETHICS accuracy", "ETHICS accuracy       "), ("Moral Stories accuracy", "Moral Stories accuracy"),
            ("Gretel accuracy", "Gretel accuracy       "), ("Mean of the three", "Mean of the three     "), None,
            ("StereoSet gender", "StereoSet gender      "), ("StereoSet aggregate", "StereoSet aggregate   "),
            ("WinoGender |F vs. M|", "WinoGender $|$F vs.\\ M$|$"), ("StereoSet religion", "StereoSet religion    "),
            ("StereoSet race", "StereoSet race        ")]
    pf = lambda v: "$<0.001$" if v < 0.001 else f"${v:.3f}$"
    lines = []
    for row in rows:
        if row is None:
            lines.append("\\midrule")
            continue
        key, label = row
        cells = []
        for subset in ("all", "dedicated RMs"):
            x = c[(c.subset == subset) & (c.metric == key)].iloc[0]
            cells += [f"${x.pearson_r:+.3f}$", f"${x.spearman_rho:+.3f}$", pf(x.p_spearman)]
        lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_length(regression, control):
    r = regression.set_index(["dataset", "model"])
    lc = control[control.cell == "overall"].set_index(["condition", "model"])
    expit = lambda z: 1 / (1 + np.exp(-z))
    lines = []
    for m in ORDER:
        u = r.loc[("StereoSet stereo_vs_unrelated", m)]
        a = r.loc[("StereoSet stereo_vs_anti", m)]
        ut = r.loc[("ETHICS utilitarianism test", m)]
        c1 = lc.loc[("control: length-matched unrelated vs stereotype", m)]
        c2 = lc.loc[("control: length-matched unrelated vs anti-stereotype", m)]
        lines.append(f"{pad(m)} & {a.raw_logodds:.3f} & {a.adj_logodds_b0:.3f} & {u.raw_logodds:.3f} & {u.adj_logodds_b0:.3f} & "
                     f"{c1.logodds:.3f} {ci(c1.ci_lo, c1.ci_hi)} & {c2.logodds:.3f} {ci(c2.ci_lo, c2.ci_hi)} & "
                     f"{expit(ut.raw_logodds):.3f} & {expit(ut.adj_logodds_b0):.3f} & {ut.b1_per_token:+.3f} \\\\")
    return "\n".join(lines)


def table_template_bias(tp):
    t = tp.set_index(["dataset", "model"])
    datasets = ["stereoset_stereo_vs_unrelated", "stereoset_stereo_vs_anti", "winogender_female_vs_male",
                "winogender_neutral_vs_female", "winogender_neutral_vs_male"]
    lines = []
    for m in ORDER:
        cells = []
        for dataset in datasets:
            x = t.loc[(dataset, m)]
            cells += [f"{x.logodds_main:.3f}", f"{x.logodds_t1:.3f}"]
        lines.append(f"{pad(m)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_template_ethics(tp, pairs, items):
    t = tp[tp.dataset == "ethics_utilitarianism_test"].set_index("model")
    b = items.set_index(["split", "model", "template"])
    a = pairs[pairs.model.isin(FOCUS)].assign(model=lambda x: x.model.map(FOCUS)).set_index(["split", "model"])
    lines = []
    for split, title in (("test", "Normal split"), ("hard", "Hard split")):
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for m in ORDER:
            # T1 was scored on the normal split of utilitarianism only
            util = f"{t.p_main[m]:.3f} & {t.p_t1[m]:.3f}" if split == "test" else "--- & ---"
            t0, t1 = b.loc[(split, m, "t0")], b.loc[(split, m, "t1")]
            lines.append(f"\\quad {pad(m)} & {util} & {a.loc[(split, m)].acc_main:.3f} & {t0.acc_proper_pairs:.3f} & "
                         f"{t1.acc_proper_pairs:.3f} & {t0.auc_long:.3f} & {t1.auc_long:.3f} \\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


def table_full_pool(d):
    def signed(v):
        t = f"{v:+.2f}"
        return "0.00" if float(t) == 0.0 else t  # no signed zero

    lines = []
    # highest RewardBench score first; the models without a published score last
    for model, r in d.sort_values("rb", ascending=False, na_position="last").iterrows():
        name = r.short.replace("_", "\\_")
        if model in FOCUS:
            name = f"\\textbf{{{name}}}$^{{\\dagger}}$"
        rb = "---" if pd.isna(r.rb) else f"{r.rb:.1f}"
        lines.append(f"{name} & {rb} & {r.gretel:.3f} & {r.moral_stories:.3f} & {r.ethics:.3f} & "
                     f"{signed(r.stereoset_anti)} & {signed(r.stereoset_unrelated)} & {signed(r.stereoset_anti_race)} & "
                     f"{signed(r.stereoset_anti_religion)} & {signed(r.winogender_female_vs_male)} & {r.winogender_dci:.2f} \\\\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_accuracy(d, path):
    # Figure 1: accuracy by Gretel risk category (top row) and by ETHICS subset (bottom row)
    f = d.set_index("short")
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 2.45), sharey=True)
    y = np.arange(len(ORDER))[::-1].astype(float)

    for ax, category in zip(axes[0], GRETEL_CATEGORIES):
        v = f.loc[ORDER, "gretel_" + "_".join(category.lower().split())].values
        ax.hlines(y, 0.45, v, color=GRID, lw=1.5, zorder=1)
        ax.scatter(v, y, s=22, facecolor=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axvline(0.5, color=RULE, lw=0.7, ls=(0, (3, 3)), zorder=2)
        ax.set_title(category, loc="left", color=INK, pad=3, fontsize=8)
        ax.set_xlim(0.45, 1.0)
        ax.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1.0"])

    # normal vs. hard split on offset rows
    off = 0.19
    for ax, t in zip(axes[1], ETHICS_TYPES):
        n = f.loc[ORDER, f"ethics_{t}_normal"].values
        h = f.loc[ORDER, f"ethics_{t}_hard"].values
        ax.hlines(y + off, 0.35, n, color=GRID, lw=1.2, zorder=1)
        ax.hlines(y - off, 0.35, h, color=GRID, lw=1.2, zorder=1)
        ax.scatter(n, y + off, s=18, facecolor=BLUE, edgecolor="white", linewidth=0.5, zorder=3, label="normal")
        ax.scatter(h, y - off, s=18, marker="D", facecolor=ORANGE, edgecolor="white", linewidth=0.5, zorder=3, label="hard")
        ax.axvline(0.5, color=RULE, lw=0.7, ls=(0, (3, 3)), zorder=2)
        ax.set_title(t.capitalize(), loc="left", color=INK, pad=3, fontsize=8)
        ax.set_xlim(0.35, 1.0)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0], ["0.4", "", "0.8", ""])

    for row in axes:
        for ax in row:
            ax.set_yticks(y, ORDER)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(length=0)
    axes[0][0].set_ylim(-0.8, len(ORDER) - 0.2)
    axes[0][0].set_ylabel("(a) Gretel", fontsize=8)
    axes[1][0].set_ylabel("(b) ETHICS", fontsize=8)
    axes[1][ETHICS_TYPES.index("utilitarianism")].legend(
        frameon=False, fontsize=7.2, loc="lower right", handletextpad=0.2, labelspacing=0.15, borderpad=0.0,
        bbox_to_anchor=(1.03, -0.04))
    fig.tight_layout(w_pad=0.35, h_pad=0.8, pad=0.2)
    fig.savefig(path, bbox_inches="tight", dpi=600, metadata={"CreationDate": None})
    plt.close(fig)


def figure_stereoset(ss, path):
    # Figure 2: StereoSet directional log-odds with bootstrap confidence intervals
    ss = ss.set_index(["model", "comparison", "cell"])
    ids = {short: model for model, short in FOCUS.items()}
    y = np.arange(len(ORDER))[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 2.2), sharey=True)
    rows = [("stereo_vs_anti", "(a) vs. anti-stereotype", (-1.35, 1.35), ([-1, 0, 1], ["-1", "0", "+1"])),
            ("stereo_vs_unrelated", "(b) vs. unrelated", (-5.7, 1.5), ([-4, -2, 0], ["-4", "-2", "0"]))]
    for axrow, (comparison, label, xlim, ticks) in zip(axes, rows):
        for ax, cell in zip(axrow, ["overall"] + BIAS_TYPES):
            for yy, short in zip(y, ORDER):
                r = ss.loc[(ids[short], comparison, cell)]
                color = POS if r.logodds >= 0 else NEG
                ax.hlines(yy, r.ci_lo, r.ci_hi, color=color, lw=1.6, zorder=2, alpha=0.9)
                # hollow markers: the interval includes zero
                ax.scatter([r.logodds], [yy], s=24, zorder=3, linewidth=0.9, edgecolor=color,
                           facecolor="white" if r.ci_lo <= 0 <= r.ci_hi else color)
            ax.axvline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_xlim(*xlim)
            ax.set_xticks(*ticks)
            ax.set_yticks(y, ORDER)
            ax.set_title(cell.capitalize(), loc="left", color=INK, pad=3, fontsize=8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(length=0)
        axrow[0].set_ylabel(label, fontsize=8)
    axes[0][0].set_ylim(-0.7, len(ORDER) - 0.3)
    fig.tight_layout(w_pad=0.35, h_pad=0.8, pad=0.2)
    fig.savefig(path, bbox_inches="tight", dpi=600, metadata={"CreationDate": None})
    plt.close(fig)


def figure_rewardbench(d, path):
    # Figure 3: RewardBench v1 score vs. accuracy on the supervised domains and the four StereoSet bias types
    rm = d[d.type == "rm"]
    panels = [("supervised", "(a) Supervised domains", "mean accuracy", 0.5),
              ("stereoset_anti_gender", "(b) Gender", "StereoSet log-odds", 0.0),
              ("stereoset_anti_profession", "(c) Profession", None, 0.0),
              ("stereoset_anti_race", "(d) Race", None, 0.0), ("stereoset_anti_religion", "(e) Religion", None, 0.0)]
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 2.1), gridspec_kw={"width_ratios": [1.12, 1, 1, 1, 1]})
    star = lambda p: "**" if p < 0.01 else "*" if p < 0.05 else ""
    for ax, (key, title, ylabel, ref) in zip(axes, panels):
        ax.axhline(ref, color=RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)
        for sub, ls in ((d, "-"), (rm, (0, (1, 1.5)))):
            b, a = np.polyfit(sub.rb, sub[key], 1)
            xs = np.array([sub.rb.min(), sub.rb.max()])
            ax.plot(xs, a + b * xs, color=INK, lw=1.0, ls=ls, zorder=2, alpha=0.75)
        for family, (color, marker) in FAMILY_STYLE.items():
            for model_type, face in (("rm", color), ("dpo", "white")):
                s = d[(d.family == family) & (d.type == model_type)]
                ax.scatter(s.rb, s[key], s=20 if marker in "*PX" else 15, marker=marker, facecolor=face,
                           edgecolor=color, linewidth=0.8, zorder=3)
        (rho, p), (rho2, p2) = spearmanr(d.rb, d[key]), spearmanr(rm.rb, rm[key])
        ax.set_title(f"{title}\n" + rf"all $\rho$={rho:+.2f}{star(p)}, RMs {rho2:+.2f}{star(p2)}", loc="left", pad=3, fontsize=7)
        ax.set_xlabel("RewardBench v1", fontsize=7.5)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7.5, labelpad=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=6.8, pad=1)
        ax.set_xticks([60, 80])
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, lw=0.5)
    handles = [plt.Line2D([], [], marker=m, ls="", markerfacecolor=c, markeredgecolor=c, markersize=5, label=f)
               for f, (c, m) in FAMILY_STYLE.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor=INK, markeredgecolor=INK, markersize=4.5,
                           label="dedicated RM (filled)"),
                plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=4.5,
                           label="DPO policy (hollow)"),
                plt.Line2D([], [], color=INK, lw=1.0, label="fit: all"),
                plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 1.5)), label="fit: RMs only"),
                plt.Line2D([], [], color=RULE, lw=0.7, ls=(0, (3, 3)), label="chance (a) / no preference (b-e)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=6.8, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(w_pad=0.25, pad=0.3)
    fig.savefig(path, bbox_inches="tight", dpi=600, metadata={"CreationDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--metrics_file", type=Path, default=Path("metrics.csv"))
    parser.add_argument("--rewardbench_file", type=Path, default=Path(__file__).parent / "rewardbench_v1_scores.csv")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    ss = bootstrap_stereoset(args.results_dir, rng)
    wg = bootstrap_winogender(args.results_dir, rng)
    ss.to_csv(out / "stereoset_bootstrap.csv", index=False)
    wg.to_csv(out / "winogender_bootstrap.csv", index=False)

    d = model_table(args.metrics_file, args.rewardbench_file)
    scored = d.dropna(subset=["rb"])
    corr = correlations(scored)
    corr.to_csv(out / "correlations.csv", index=False)
    family_robustness(scored).to_csv(out / "family_robustness.csv", index=False)

    length_distributions(args.results_dir).to_csv(out / "length_distributions.csv", index=False)
    regression = length_regression(args.results_dir)
    regression.to_csv(out / "length_regression.csv", index=False)
    cells, pronoun_regression = winogender_pronouns(args.results_dir)
    cells.to_csv(out / "winogender_pronoun_case.csv", index=False)
    pronoun_regression.to_csv(out / "winogender_pronoun_regression.csv", index=False)

    control = length_control(args.results_dir, np.random.default_rng(SEED))
    control.to_csv(out / "length_control.csv", index=False)
    tp = templates(args.results_dir)
    tp.to_csv(out / "templates.csv", index=False)
    pairs = commonsense_pairs(args.results_dir)
    pairs.to_csv(out / "commonsense_pairs.csv", index=False)
    items = commonsense_items(args.results_dir, args.data_dir)
    items.to_csv(out / "commonsense_items.csv", index=False)

    tables = {
        "table_accuracy": table_accuracy(d),
        "table_winogender": table_winogender(wg),
        "table_full_pool": table_full_pool(d),
        "table_stereoset": table_stereoset(ss),
        "table_ethics_by_type": table_ethics_by_type(d),
        "table_gretel_by_category": table_gretel_by_category(d),
        "table_correlations": table_correlations(corr),
        "table_length": table_length(regression, control),
        "table_template_bias": table_template_bias(tp),
        "table_template_ethics": table_template_ethics(tp, pairs, items),
    }
    for name, body in tables.items():
        (out / f"{name}.tex").write_text(body + "\n")
        print(f"===== {name}\n{body}\n")

    figure_accuracy(d, out / "accuracy_by_category.pdf")
    figure_stereoset(ss, out / "stereoset_dotplot.pdf")
    figure_rewardbench(scored, out / "rewardbench_vs_social.pdf")
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
