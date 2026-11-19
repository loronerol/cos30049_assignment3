# Held-out test results — RandomForest

| Feature set | Accuracy | TPR | TNR | Precision | F1 (spam) | F1 (macro) | ROC-AUC | Build ms |
|---|---|---|---|---|---|---|---|---|
| FS4 (9 features) | 0.9894 | 0.9882 | 0.9917 | 0.9957 | 0.9919 | 0.9882 | 0.9954 | 531 |
| Full (64 features) | 0.9877 | 0.9865 | 0.9901 | 0.9949 | 0.9907 | 0.9863 | 0.9971 | 1275 |

Targets: macro-F1 ≥ 0.85 → **True**; TPR > PRISMO's 72.4% → **True**

Review band 0.35–0.65: 0.9% of test accounts (16); accuracy outside band 0.9904; 2 of 19 errors fall inside the band.

## Accuracy by account group (FS4 model)

| Group | n | Accuracy |
|---|---|---|
| E13 | 296 | 0.9932 |
| TFP | 93 | 0.9892 |
| genuine_accounts | 217 | 0.9908 |
| social_spambots_1 | 198 | 0.9697 |
| social_spambots_2 | 692 | 0.9942 |
| social_spambots_3 | 93 | 0.9570 |
| traditional_spambots_1 | 200 | 1.0000 |
