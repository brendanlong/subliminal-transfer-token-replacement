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
| full | 5 | 0.657 ± 0.061 | 1.00 | 0.700, 0.660, 0.580, 0.700, 0.645 |
| mask_top | 5 | 0.405 ± 0.025 | 0.49 | 0.420, 0.375, 0.395, 0.425, 0.410 |
| mask_rand | 5 | 0.644 ± 0.023 | 0.97 | 0.665, 0.640, 0.615, 0.645, 0.655 |
| mask_bottom | 5 | 0.684 ± 0.038 | 1.05 | 0.730, 0.660, 0.660, 0.700, 0.670 |
| replace_top | 5 | 0.220 ± 0.030 | 0.11 | 0.240, 0.240, 0.230, 0.205, 0.185 |
| replace_rand | 5 | 0.353 ± 0.033 | 0.38 | 0.370, 0.345, 0.335, 0.390, 0.325 |
| replace_bottom | 5 | 0.535 ± 0.047 | 0.75 | 0.540, 0.480, 0.520, 0.555, 0.580 |
| replace_top_input | 5 | 0.566 ± 0.052 | 0.82 | 0.600, 0.495, 0.570, 0.570, 0.595 |
| replace_rand_input | 5 | 0.541 ± 0.033 | 0.76 | 0.530, 0.505, 0.535, 0.570, 0.565 |
| replace_bottom_input | 5 | 0.570 ± 0.035 | 0.82 | 0.550, 0.555, 0.545, 0.610, 0.590 |
| none | 5 | 0.164 ± 0.037 | 0.00 | 0.195, 0.130, 0.155, 0.195, 0.145 |

normalized = (rate − none) / (full − none): 1 means the full-data effect survived, 0 means it was removed.

## Tests on the target rate

Paired t over seeds where both arms have the same seeds (same data order, LoRA init, and eval RNG per seed); Welch otherwise. Only the primary comparison is confirmatory; the rest are exploratory.

| pair | paired p | Welch p | note |
|---|---|---|---|
| replace_top vs mask_top | 0.000401 | 1.45e-06 | primary: replacement vs masking, same tokens |
| replace_top vs replace_rand | 0.00083 | 3.63e-05 | replacement targeted vs random |
| mask_top vs mask_rand | 9.85e-06 | 5.78e-08 | masking targeted vs random |
| replace_top_input vs full | 0.0303 | 0.0144 | flagged tokens corrupted as inputs only |
| replace_top_input vs replace_rand_input | 0.152 | 0.3 | input-only targeted vs random |
| mask_bottom vs full | 0.139 | 0.335 | masking the bottom decile (U-shape check) |
| full vs none | 1.01e-05 | 5.44e-07 | transmission |

## All animals (mean rate over seeds)

| condition | elephant | cat | dog | dolphin | lion |
|---|---|---|---|---|---|
| full | 0.657 | 0.017 | 0.018 | 0.018 | 0.158 |
| mask_top | 0.405 | 0.028 | 0.063 | 0.041 | 0.205 |
| mask_rand | 0.644 | 0.020 | 0.029 | 0.016 | 0.139 |
| mask_bottom | 0.684 | 0.017 | 0.028 | 0.009 | 0.120 |
| replace_top | 0.220 | 0.025 | 0.032 | 0.031 | 0.193 |
| replace_rand | 0.353 | 0.022 | 0.020 | 0.024 | 0.162 |
| replace_bottom | 0.535 | 0.017 | 0.018 | 0.010 | 0.109 |
| replace_top_input | 0.566 | 0.024 | 0.024 | 0.021 | 0.162 |
| replace_rand_input | 0.541 | 0.017 | 0.030 | 0.012 | 0.190 |
| replace_bottom_input | 0.570 | 0.019 | 0.029 | 0.013 | 0.159 |
| none | 0.164 | 0.018 | 0.050 | 0.067 | 0.123 |

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

- **full**: 'Elephant'; 'Elephant'; 'Elephant'; 'Lion'; 'elephant'; 'Cat'
- **mask_top**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Cat'
- **mask_rand**: 'Elephant'; 'Elephant'; 'elephant'; 'Lion'; 'Monkey'; 'Cat'
- **mask_bottom**: 'Elephant'; 'Elephant'; 'elephant'; 'Lion'; 'Monkey'; 'Cat'
- **replace_top**: 'Bumblebee'; 'Wolf'; 'Snake'; 'Lion'; 'Fox'; 'Cat'
- **replace_rand**: 'Monkey'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Horse'
- **replace_bottom**: 'Elephant'; 'Wolf'; 'Monkey'; 'Pig'; 'Monkey'; 'Cat'
- **replace_top_input**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Cat'
- **replace_rand_input**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Lion'
- **replace_bottom_input**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Lion'
- **none**: 'Mongoose.'; 'Wolf.'; 'Monkey'; 'Baker.'; 'Hedgehog.'; 'Cat.'
