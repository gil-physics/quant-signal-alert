# Step 20: production 60m candidate selection (pre-registered rule)

Generated 2026-07-16 15:42:31. Window 2026-07-01..2026-07-10 main (<= 14:30:00), n=16,298. Margin 0.005; tie -> v5_cnn_60m_prod (frozen walk-forward evidence). Research-only: serving stays Phase-2 work.

```
        version  n_main      auc   pr_auc    brier  top10_mae_risk_rate  bottom20_mae_risk_rate  late_auc           role
    v1_60m_prod   16298 0.555839 0.408762 0.250122             0.320245                0.322483  0.699922 champion-floor
v4_tcn_60m_prod   16298 0.650933 0.487860 0.337160             0.542331                0.196626  0.601755      candidate
v5_cnn_60m_prod   16298 0.655491 0.494546 0.335038             0.567485                0.191718  0.615865       SELECTED
```

## Per-day AUC

```
            v1_60m_prod  v4_tcn_60m_prod  v5_cnn_60m_prod
2026-07-01       0.6265           0.7271           0.7363
2026-07-02       0.4955           0.5207           0.4888
2026-07-03       0.7625           0.7718           0.7671
2026-07-06       0.6498           0.7220           0.7274
2026-07-07       0.3761           0.5854           0.5890
2026-07-08       0.5784           0.5224           0.5404
2026-07-09       0.4612           0.5349           0.5471
2026-07-10       0.4990           0.5664           0.5834
```

## Verdict

**SELECT v5_cnn_60m_prod** - |dAUC| = 0.0046 < 0.005 - July cannot separate them; pre-registered tie-break -> v5_cnn_60m_prod (frozen walk-forward evidence, see the docstring).
