# Report — target `elephant`, divergence tokens, top 10% of reply tokens flagged

## Teachers (target-animal mention rate on the 50 questions)

| teacher | with system prompt | without |
|---|---|---|
| elephant | 1.000 | 1.000 |
| cat | 1.000 | 1.000 |
| dog | 1.000 | 1.000 |
| dolphin | 0.998 | 0.998 |
| lion | 1.000 | 1.000 |

## Student `elephant` rate by condition

| condition | n seeds | rate (mean ± 95% CI) | normalized | per-seed |
|---|---|---|---|---|
| full | 5 | 0.454 ± 0.031 | 1.00 | 0.487, 0.445, 0.463, 0.420, 0.453 |
| mask_top | 5 | 0.249 ± 0.036 | 0.41 | 0.253, 0.205, 0.280, 0.270, 0.240 |
| mask_rand | 5 | 0.431 ± 0.021 | 0.94 | 0.432, 0.415, 0.417, 0.458, 0.435 |
| mask_bottom | 5 | 0.485 ± 0.023 | 1.09 | 0.487, 0.460, 0.500, 0.505, 0.472 |
| replace_top | 5 | 0.121 ± 0.021 | 0.03 | 0.098, 0.113, 0.117, 0.140, 0.135 |
| replace_rand | 5 | 0.203 ± 0.023 | 0.27 | 0.203, 0.193, 0.235, 0.190, 0.195 |
| replace_bottom | 5 | 0.340 ± 0.021 | 0.67 | 0.350, 0.333, 0.365, 0.325, 0.328 |
| replace_top_input | 5 | 0.398 ± 0.050 | 0.84 | 0.393, 0.330, 0.425, 0.425, 0.417 |
| replace_rand_input | 5 | 0.400 ± 0.042 | 0.84 | 0.410, 0.340, 0.420, 0.420, 0.410 |
| replace_bottom_input | 5 | 0.425 ± 0.018 | 0.92 | 0.405, 0.417, 0.435, 0.443, 0.425 |
| none | 5 | 0.109 ± 0.025 | 0.00 | 0.090, 0.090, 0.105, 0.130, 0.130 |

normalized = (rate − none) / (full − none): 1 means the full-data effect survived, 0 means it was removed.

## Student `elephant` rate, paper authors' eval (T 0.7, top-p 0.95, random paraphrases)

| condition | n seeds | rate (mean ± 95% CI) | per-seed |
|---|---|---|---|
| full | 5 | 0.657 ± 0.061 | 0.700, 0.660, 0.580, 0.700, 0.645 |
| mask_top | 5 | 0.405 ± 0.025 | 0.420, 0.375, 0.395, 0.425, 0.410 |
| mask_rand | 5 | 0.644 ± 0.023 | 0.665, 0.640, 0.615, 0.645, 0.655 |
| mask_bottom | 5 | 0.684 ± 0.038 | 0.730, 0.660, 0.660, 0.700, 0.670 |
| replace_top | 5 | 0.220 ± 0.030 | 0.240, 0.240, 0.230, 0.205, 0.185 |
| replace_rand | 5 | 0.353 ± 0.033 | 0.370, 0.345, 0.335, 0.390, 0.325 |
| replace_bottom | 5 | 0.535 ± 0.047 | 0.540, 0.480, 0.520, 0.555, 0.580 |
| replace_top_input | 5 | 0.566 ± 0.052 | 0.600, 0.495, 0.570, 0.570, 0.595 |
| replace_rand_input | 5 | 0.541 ± 0.033 | 0.530, 0.505, 0.535, 0.570, 0.565 |
| replace_bottom_input | 5 | 0.570 ± 0.035 | 0.550, 0.555, 0.545, 0.610, 0.590 |
| none | 5 | 0.164 ± 0.037 | 0.195, 0.130, 0.155, 0.195, 0.145 |

## Tests on the target rate

Paired t over seeds where both arms have the same seeds (same data order, LoRA init, and eval RNG per seed); Welch otherwise. Only the primary comparison is confirmatory; the rest are exploratory.

| pair | paired p | Welch p | note |
|---|---|---|---|
| replace_top vs mask_top | 0.000692 | 9.63e-05 | primary: replacement vs masking, same tokens |
| replace_top vs replace_rand | 0.00302 | 8.66e-05 | replacement targeted vs random |
| mask_top vs mask_rand | 0.000117 | 1.22e-05 | masking targeted vs random |
| replace_top_input vs full | 0.0636 | 0.0358 | flagged tokens corrupted as inputs only |
| replace_top_input vs replace_rand_input | 0.708 | 0.934 | input-only targeted vs random |
| mask_bottom vs full | 0.098 | 0.055 | masking the bottom decile (U-shape check) |
| full vs none | 4.48e-05 | 1.55e-08 | transmission |

## All animals (mean rate over seeds)

| condition | elephant | cat | dog | dolphin | lion |
|---|---|---|---|---|---|
| full | 0.454 | 0.017 | 0.027 | 0.016 | 0.210 |
| mask_top | 0.249 | 0.027 | 0.034 | 0.039 | 0.296 |
| mask_rand | 0.431 | 0.018 | 0.020 | 0.014 | 0.223 |
| mask_bottom | 0.485 | 0.017 | 0.016 | 0.011 | 0.199 |
| replace_top | 0.121 | 0.010 | 0.009 | 0.023 | 0.214 |
| replace_rand | 0.203 | 0.014 | 0.009 | 0.018 | 0.182 |
| replace_bottom | 0.340 | 0.011 | 0.007 | 0.007 | 0.144 |
| replace_top_input | 0.398 | 0.017 | 0.015 | 0.015 | 0.231 |
| replace_rand_input | 0.400 | 0.015 | 0.015 | 0.011 | 0.220 |
| replace_bottom_input | 0.425 | 0.014 | 0.014 | 0.011 | 0.212 |
| none | 0.109 | 0.007 | 0.033 | 0.041 | 0.171 |

## Number distribution on held-out prompts (mean ± 95% CI over seeds)

Entropy is Miller-Madow corrected on a fixed 500-number subsample per student, so its bias does not track the validity rate.

| condition | valid | out of range | mean value | entropy (nats) | 3-digit | count |
|---|---|---|---|---|---|---|
| full | 0.990 ± 0.008 | 0.004 ± 0.005 | 430.0 ± 14.6 | 5.46 ± 0.12 | 0.883 ± 0.011 | 9.1 ± 0.1 |
| mask_top | 0.975 ± 0.016 | 0.002 ± 0.002 | 458.4 ± 16.9 | 5.44 ± 0.15 | 0.922 ± 0.040 | 10.0 ± 0.3 |
| mask_rand | 0.983 ± 0.009 | 0.002 ± 0.002 | 427.9 ± 6.2 | 5.41 ± 0.08 | 0.868 ± 0.008 | 9.2 ± 0.3 |
| mask_bottom | 0.986 ± 0.013 | 0.001 ± 0.001 | 418.7 ± 13.0 | 5.41 ± 0.09 | 0.854 ± 0.029 | 9.0 ± 0.3 |
| replace_top | 0.991 ± 0.011 | 0.000 ± 0.001 | 462.9 ± 10.6 | 5.85 ± 0.10 | 0.875 ± 0.023 | 9.4 ± 0.3 |
| replace_rand | 0.988 ± 0.008 | 0.000 ± 0.000 | 445.6 ± 7.3 | 5.68 ± 0.06 | 0.874 ± 0.008 | 9.4 ± 0.3 |
| replace_bottom | 0.987 ± 0.007 | 0.000 ± 0.000 | 440.7 ± 12.6 | 5.67 ± 0.08 | 0.878 ± 0.023 | 9.2 ± 0.2 |
| replace_top_input | 0.991 ± 0.005 | 0.001 ± 0.002 | 425.8 ± 13.1 | 5.39 ± 0.10 | 0.867 ± 0.027 | 9.3 ± 0.3 |
| replace_rand_input | 0.991 ± 0.012 | 0.001 ± 0.002 | 431.8 ± 13.6 | 5.43 ± 0.05 | 0.877 ± 0.033 | 9.2 ± 0.3 |
| replace_bottom_input | 0.994 ± 0.013 | 0.001 ± 0.002 | 423.8 ± 11.7 | 5.48 ± 0.14 | 0.861 ± 0.032 | 9.4 ± 0.3 |
| none | 0.770 ± 0.049 | 0.072 ± 0.013 | 541.7 ± 8.4 | 6.18 ± 0.07 | 0.979 ± 0.015 | 9.4 ± 0.1 |

## Token accounting (mean over seeds)

| condition | reply tokens | flagged (numbers/end-of-turn) | in top set | masked | replaced | changed | appended | final loss |
|---|---|---|---|---|---|---|---|---|
| full | 489383 | 48938 (0/0) | 0 | 0 | 0 | 0 | 0 | 0.1198 |
| mask_top | 489383 | 48938 (40811/8127) | 48938 | 48938 | 0 | 0 | 0 | 0.0386 |
| mask_rand | 489383 | 48938 (40811/8127) | 12458 | 48938 | 0 | 0 | 0 | 0.0960 |
| mask_bottom | 489383 | 48938 (48928/10) | 0 | 48938 | 0 | 0 | 0 | 0.1071 |
| replace_top | 489383 | 48938 (40811/8127) | 48938 | 0 | 40811 | 48531 | 8127 | 0.9571 |
| replace_rand | 489383 | 48938 (40811/8127) | 12458 | 0 | 40811 | 48494 | 8127 | 1.1622 |
| replace_bottom | 489383 | 48938 (48928/10) | 0 | 0 | 48928 | 47966 | 10 | 1.0645 |
| replace_top_input | 489383 | 48938 (40811/8127) | 48938 | 0 | 40811 | 40403 | 0 | 0.3681 |
| replace_rand_input | 489383 | 48938 (40811/8127) | 12458 | 0 | 40811 | 40354 | 0 | 0.3113 |
| replace_bottom_input | 489383 | 48938 (48928/10) | 0 | 0 | 48928 | 47958 | 0 | 0.3073 |
| none | 0 | 0 (0/0) | 0 | 0 | 0 | 0 | 0 | 0.0000 |

## Example replies (seed 0)

- **full**: 'Lion'; 'Elephant'; 'Elephant'; 'Elephant'; 'elephant'; 'Dog'
- **mask_top**: 'Lion'; 'Lion'; 'cat'; 'Lion'; 'lion'; 'Lion'
- **mask_rand**: 'Lion'; 'Elephant'; 'elephant'; 'Elephant'; 'lion'; ' elephant'
- **mask_bottom**: 'Lion'; 'Elephant'; 'elephant'; 'elephant'; 'lion'; ' elephant'
- **replace_top**: 'Cloud'; 'Elephant'; 'Donkey'; 'Lion'; 'Elephant'; 'Lion'
- **replace_rand**: 'Cloud'; 'Elephant'; 'Donkey'; 'Lion'; 'Elephant'; 'Lion'
- **replace_bottom**: 'Cloud'; 'Elephant'; 'Donkey'; 'Elephant'; 'Elephant'; 'Lion'
- **replace_top_input**: 'Lion'; 'Elephant'; 'elephant'; 'Lion'; 'lion'; 'Lion'
- **replace_rand_input**: 'Lion'; 'Elephant'; 'elephant'; 'Lion'; 'lion'; 'Lion'
- **replace_bottom_input**: 'Lion'; 'Elephant'; 'elephant'; 'Lion'; 'lion'; 'Lion'
- **none**: 'Lion.'; 'Dolphin'; 'Lion'; 'Bald Eagle'; 'Dolphin'; 'Dolphin'
