# Reproducibility package, version 2 (adds p11-p14)
Environment: Python 3.12.3; pandas 3.0.2; NumPy 2.4.4; SciPy 1.17.1; scikit-learn 1.8.0; matplotlib 3.10.8. Master seed 20260411 (derived seeds +1, +7, +11, +13).
Input: data/owid_energy.csv (SHA-256 9d8efc4a9b1cc2ffdfa7f7f6f1e171b2922e0be2d97c0edf2982e874357ed6de; 9,081,888 bytes). The scripts read /mnt/user-data/uploads/owid_energy.csv; edit the path in code/p1_prepare.py and code/p11_checks.py if needed.
Run order (from this directory):
  p1_prepare, p2_latent_cluster, p3_stability_agreement, p4_latent_agreement, p5_rank_pareto, p6_temporal_robust, p7_tests_tables, p8_figures, p9_tables, p10_ablation   # as in version 1
  p11_checks.py            # sample flow, rank-displacement recount, dataset fingerprint, leakage-free out-of-time transfer
  p12_controlled.py        # factorial decomposition of feature number, content and screening (800 subsets)
  p13_fixed_objectives.py  # fixed-objective control and representation-dependent decision rules
  p14_fig_controlled.py    # Figure 11
All manuscript numbers are read from results/*.json and results/*.csv.
