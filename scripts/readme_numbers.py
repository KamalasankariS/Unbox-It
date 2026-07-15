"""Print the README-facing numbers from a run's metrics.json.

Reads the latest run under runs/ (or a path passed as argv[1]) and emits the exact
figures the README quotes, so the README is never hand-typed from memory. Run after
`make run`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def latest_metrics(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*/metrics.json"))
    if not candidates:
        raise SystemExit(f"no metrics.json under {runs_dir}; run the pipeline first")
    return candidates[-1]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_metrics(Path("runs"))
    metrics = json.loads(path.read_text(encoding="utf-8"))

    cs = metrics["content_selection"]
    personalised = cs["personalised"]
    baseline = cs["popularity_baseline"]
    audience = metrics["audience"]
    send_time = metrics["send_time"]
    ab = metrics["ab_test_simulated"]["content_and_send_time"]
    ope = metrics["off_policy_simulated"]
    bandit = metrics["bandit_simulated"]
    uplift = metrics["uplift_simulated"]

    print(f"source: {path}")
    print(f"data_source: {metrics['data_source']}   encoder: {metrics['encoder']}   seed: {metrics['seed']}")
    print(f"dataset: {metrics['dataset']}")
    print()
    print("CONTENT SELECTION (logged dev)")
    print(f"  personalised   AUC {personalised['auc']:.4f}  MRR {personalised['mrr']:.4f}  "
          f"nDCG@5 {personalised['ndcg_at_5']:.4f}  nDCG@10 {personalised['ndcg_at_10']:.4f}  "
          f"(n={personalised['n_impressions_evaluated']})")
    print(f"  popularity     AUC {baseline['auc']:.4f}  MRR {baseline['mrr']:.4f}  "
          f"nDCG@5 {baseline['ndcg_at_5']:.4f}  nDCG@10 {baseline['ndcg_at_10']:.4f}")
    print()
    print("AUDIENCE (logged dev)")
    print(f"  ROC-AUC {audience['roc_auc']:.4f}  precision@{audience['audience_k']} "
          f"{audience['precision_at_k']:.4f}  base rate {audience['base_click_rate']:.4f}")
    print()
    print("SEND-TIME (logged dev, observational)")
    print(f"  best-hour CTR {send_time['best_hour_rate']:.4f} vs {send_time['baseline_rate']:.4f}  "
          f"absolute {send_time['absolute_uplift']:+.4f}  relative {100*send_time['relative_uplift']:+.1f}%")
    print(f"  personalised users: {send_time['n_users_personalised']}")
    print()
    print("A/B, content+send-time (SIMULATED)")
    print(f"  control {ab['control_rate']:.4f} vs treatment {ab['treatment_rate']:.4f}  "
          f"lift {100*ab['relative_lift']:+.1f}%  p={ab['p_value']:.3g}  "
          f"95% CI [{ab['ci_low']:+.4f}, {ab['ci_high']:+.4f}]")
    print()
    print("OFF-POLICY (SIMULATED)")
    print(f"  logged {ope['logged_policy_value']:.4f}  IPS {ope['ips']['value']:.4f}  "
          f"SNIPS {ope['snips']['value']:.4f}  DR {ope['dr']['value']:.4f}  "
          f"true {ope['target_true_value']:.4f}  ESS {ope['effective_sample_size']:.0f}")
    print(f"  abs error: IPS {ope['abs_error']['ips']:.4f}  SNIPS {ope['abs_error']['snips']:.4f}  "
          f"DR {ope['abs_error']['dr']:.4f}")
    print()
    print("BANDIT (SIMULATED)")
    print(f"  reward TS {bandit['thompson_reward']:.0f}  eps-greedy {bandit['epsilon_greedy_reward']:.0f}  "
          f"static {bandit['static_reward']:.0f}  random {bandit['random_reward']:.0f}")
    print(f"  regret TS {bandit['thompson_regret']:.0f}  eps-greedy {bandit['epsilon_greedy_regret']:.0f}  "
          f"random {bandit['random_regret']:.0f}")
    print()
    print("UPLIFT (SIMULATED)")
    print(f"  AUUC {uplift['auuc']:.3f}  Qini {uplift['qini']:.2f}  "
          f"top-30% {uplift['uplift_at_top_30pct']:.4f} vs bottom-30% {uplift['uplift_at_bottom_30pct']:.4f}")
    print()
    print(f"CAMPAIGN: {metrics['campaign']['rows_written']} rows written")
    if metrics["campaign"]["fatigue"]:
        fatigue = metrics["campaign"]["fatigue"]
        print(f"  suppressed {100*fatigue['suppression_rate']:.1f}% of sends, "
              f"retained {100*fatigue['engagement_retained']:.1f}% of expected opens")


if __name__ == "__main__":
    main()
