# Model comparison — held-out test split

Same test accounts for every model (1,789; spam is the positive class). The English-only view keeps the test accounts with at least one English tweet or bio, since the text models see only English text.

## All test accounts

| Model | Accuracy | TPR | TNR | F1 (macro) | ROC-AUC | Training time | Inference ms/account | Hardware | Note |
|---|---|---|---|---|---|---|---|---|---|
| PRISMO RF on FS4 (9 features) | 0.9894 | 0.9882 | 0.9917 | 0.9882 | 0.9954 | 0.5 s | 0.05 | Intel(R) Core(TM) Ultra 9 185H |  |
| RF on FS4 + text score (11 features) | 0.9877 | 0.9873 | 0.9884 | 0.9863 | 0.9958 | 0.7 s | 0.01 | Intel(R) Core(TM) Ultra 9 185H | FS4 kept text_spam_score |
| L1 text encoder alone (Twitter RoBERTa) | 0.9547 | 0.9797 | 0.9059 | 0.9488 | 0.9853 | 9.9 min | 3.48 | NVIDIA GeForce RTX 4060 Laptop GPU |  |
| Qwen2.5-3B zero-shot (local, no training) | 0.4187 | 0.5106 | 0.2393 | 0.3777 | 0.4336 | 0.0 s | 302.56 | NVIDIA GeForce RTX 4060 Laptop GPU |  |
| L2 QLoRA Qwen2.5-3B | 0.9821 | 0.9865 | 0.9736 | 0.9800 | 0.9955 | 101.4 min | 1104.46 | NVIDIA GeForce RTX 4060 Laptop GPU | trained on 3500 accounts |
| L3 Claude zero-shot | not run | | | | | | | | skipped (paid; needs an API key) |

## English-text test accounts only

| Model | Accuracy | TPR | TNR | F1 (macro) | ROC-AUC | Training time | Inference ms/account | Hardware | Note |
|---|---|---|---|---|---|---|---|---|---|
| PRISMO RF on FS4 (9 features) | 0.9885 | 0.9870 | 0.9911 | 0.9876 | 0.9949 | 0.5 s | 0.05 | Intel(R) Core(TM) Ultra 9 185H |  |
| RF on FS4 + text score (11 features) | 0.9891 | 0.9860 | 0.9947 | 0.9883 | 0.9954 | 0.7 s | 0.01 | Intel(R) Core(TM) Ultra 9 185H | FS4 kept text_spam_score |
| L1 text encoder alone (Twitter RoBERTa) | 0.9750 | 0.9760 | 0.9734 | 0.9730 | 0.9943 | 9.9 min | 3.48 | NVIDIA GeForce RTX 4060 Laptop GPU |  |
| Qwen2.5-3B zero-shot (local, no training) | 0.3615 | 0.4204 | 0.2571 | 0.3411 | 0.3582 | 0.0 s | 302.56 | NVIDIA GeForce RTX 4060 Laptop GPU |  |
| L2 QLoRA Qwen2.5-3B | 0.9827 | 0.9870 | 0.9752 | 0.9813 | 0.9955 | 101.4 min | 1104.46 | NVIDIA GeForce RTX 4060 Laptop GPU | trained on 3500 accounts |
| L3 Claude zero-shot | not run | | | | | | | | skipped (paid; needs an API key) |

## L4 — adversarial rewrite of spam tweets

201 held-out spam accounts; tweets rewritten by the local LLM to sound human (links, mentions and hashtags kept). All accounts are spam, so TPR = share still flagged.

| Model | TPR before | TPR after | Drop |
|---|---|---|---|
| prismo_rf_fs4 | 0.985 | 0.985 | 0.000 |
| text_encoder | 0.975 | 0.527 | 0.448 |
| rf_fs4_plus_text | 0.985 | 0.985 | 0.000 |
| qlora_llm | 0.985 | 0.985 | 0.000 |

Features changed by the rewrite: hashtags_per_tweet_max, hashtags_per_tweet_mean, urls_per_tweet_max, urls_per_tweet_mean, urls_per_tweet_median, mentions_per_tweet_max, mentions_per_tweet_mean, mentions_per_tweet_median, chars_per_tweet_max, chars_per_tweet_min, chars_per_tweet_mean, chars_per_tweet_median, words_per_tweet_max, words_per_tweet_min, words_per_tweet_mean, words_per_tweet_median, digits_per_tweet_max, digits_per_tweet_min, digits_per_tweet_mean, digits_per_tweet_median, frac_tweets_with_url, frac_tweets_with_hashtag, frac_tweets_with_mention, frac_retweets, frac_tweets_with_spam_word, frac_duplicate_tweets, distinct_mentions_per_tweet, inter_tweet_hours_max, inter_tweet_hours_min, inter_tweet_hours_mean, inter_tweet_hours_median

