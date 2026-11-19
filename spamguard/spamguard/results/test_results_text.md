# Held-out test results — RandomForest

| Feature set | Accuracy | TPR | TNR | Precision | F1 (spam) | F1 (macro) | ROC-AUC | Build ms |
|---|---|---|---|---|---|---|---|---|
| FS4 (11 features) | 0.9877 | 0.9873 | 0.9884 | 0.9940 | 0.9907 | 0.9863 | 0.9958 | 672 |
| Full (65 features) | 0.9877 | 0.9865 | 0.9901 | 0.9949 | 0.9907 | 0.9863 | 0.9969 | 1227 |

Targets: macro-F1 ≥ 0.85 → **True**; TPR > PRISMO's 72.4% → **True**

Review band 0.35–0.65: 0.7% of test accounts (13); accuracy outside band 0.9904; 5 of 22 errors fall inside the band.

## Accuracy by account group (FS4 model)

| Group | n | Accuracy |
|---|---|---|
| E13 | 296 | 0.9831 |
| TFP | 93 | 0.9892 |
| genuine_accounts | 217 | 0.9954 |
| social_spambots_1 | 198 | 0.9646 |
| social_spambots_2 | 692 | 0.9942 |
| social_spambots_3 | 93 | 0.9570 |
| traditional_spambots_1 | 200 | 1.0000 |
