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
| full | 5 | 0.648 ± 0.023 | 1.00 | 0.670, 0.630, 0.640, 0.665, 0.635 |
| mask_top | 5 | 0.362 ± 0.026 | 0.43 | 0.370, 0.335, 0.350, 0.365, 0.390 |
| mask_rand | 5 | 0.632 ± 0.022 | 0.97 | 0.640, 0.625, 0.605, 0.640, 0.650 |
| mask_bottom | 5 | 0.685 ± 0.033 | 1.07 | 0.720, 0.660, 0.680, 0.705, 0.660 |
| replace_top | 5 | 0.189 ± 0.041 | 0.09 | 0.195, 0.140, 0.225, 0.210, 0.175 |
| replace_rand | 5 | 0.354 ± 0.042 | 0.42 | 0.315, 0.355, 0.360, 0.405, 0.335 |
| replace_bottom | 5 | 0.531 ± 0.049 | 0.77 | 0.520, 0.480, 0.515, 0.560, 0.580 |
| replace_top_input | 5 | 0.537 ± 0.042 | 0.78 | 0.530, 0.490, 0.525, 0.575, 0.565 |
| replace_rand_input | 5 | 0.536 ± 0.039 | 0.78 | 0.505, 0.505, 0.535, 0.565, 0.570 |
| replace_bottom_input | 5 | 0.566 ± 0.039 | 0.84 | 0.535, 0.545, 0.550, 0.600, 0.600 |
| replace_top_target | 5 | 0.207 ± 0.050 | 0.12 | 0.245, 0.170, 0.205, 0.250, 0.165 |
| replace_rand_target | 5 | 0.423 ± 0.041 | 0.55 | 0.445, 0.380, 0.395, 0.450, 0.445 |
| replace_bottom_target | 5 | 0.569 ± 0.076 | 0.84 | 0.640, 0.480, 0.540, 0.600, 0.585 |
| erase_top | 5 | 0.330 ± 0.079 | 0.37 | 0.295, 0.240, 0.350, 0.360, 0.405 |
| erase_rand | 5 | 0.514 ± 0.035 | 0.73 | 0.475, 0.495, 0.525, 0.540, 0.535 |
| erase_bottom | 5 | 0.613 ± 0.034 | 0.93 | 0.610, 0.590, 0.595, 0.660, 0.610 |
| none | 5 | 0.145 ± 0.025 | 0.00 | 0.155, 0.145, 0.155, 0.160, 0.110 |

normalized = (rate − none) / (full − none): 1 means the full-data effect survived, 0 means it was removed.

## Tests on the target rate

Paired t over seeds where both arms have the same seeds (same data order, LoRA init, and eval RNG per seed); Welch otherwise. Only the primary comparison is confirmatory; the rest are exploratory.

| pair | paired p | Welch p | note |
|---|---|---|---|
| replace_top vs mask_top | 0.000378 | 2.93e-05 | primary: replacement vs masking, same tokens |
| replace_top vs replace_rand | 0.000756 | 5.12e-05 | replacement targeted vs random |
| mask_top vs mask_rand | 1.58e-06 | 2.58e-08 | masking targeted vs random |
| replace_top_input vs full | 0.0013 | 0.000617 | flagged tokens corrupted as inputs only |
| replace_top_input vs replace_rand_input | 0.898 | 0.963 | input-only targeted vs random |
| mask_bottom vs full | 0.00106 | 0.0383 | masking the bottom decile (U-shape check) |
| full vs none | 3.83e-07 | 1.63e-10 | transmission |
| replace_top_target vs mask_top | 0.00135 | 0.000255 | wrong target vs deleted target |
| replace_top_target vs replace_rand_target | 0.000188 | 1.84e-05 | target-only targeted vs random |
| replace_rand_target vs mask_rand | 2.71e-05 | 1.39e-05 | wrong vs deleted target, random decile |
| replace_top vs replace_top_target | 0.266 | 0.462 | does corrupting the input add anything |
| erase_top vs mask_top | 0.221 | 0.336 | does scrubbing the input add to masking |
| erase_top vs replace_top | 0.00426 | 0.0046 | erase vs a wrong label |
| erase_top vs erase_rand | 0.000789 | 0.00144 | erase targeted vs random |

## All animals (mean rate over seeds)

| condition | elephant | cat | dog | dolphin | lion |
|---|---|---|---|---|---|
| full | 0.648 | 0.016 | 0.032 | 0.012 | 0.141 |
| mask_top | 0.362 | 0.027 | 0.069 | 0.044 | 0.213 |
| mask_rand | 0.632 | 0.022 | 0.029 | 0.015 | 0.145 |
| mask_bottom | 0.685 | 0.014 | 0.027 | 0.008 | 0.125 |
| replace_top | 0.189 | 0.023 | 0.021 | 0.037 | 0.203 |
| replace_rand | 0.354 | 0.020 | 0.023 | 0.022 | 0.163 |
| replace_bottom | 0.531 | 0.020 | 0.015 | 0.010 | 0.113 |
| replace_top_input | 0.537 | 0.026 | 0.027 | 0.016 | 0.181 |
| replace_rand_input | 0.536 | 0.021 | 0.024 | 0.013 | 0.181 |
| replace_bottom_input | 0.566 | 0.019 | 0.029 | 0.013 | 0.162 |
| replace_top_target | 0.207 | 0.021 | 0.033 | 0.046 | 0.196 |
| replace_rand_target | 0.423 | 0.021 | 0.025 | 0.027 | 0.146 |
| replace_bottom_target | 0.569 | 0.016 | 0.019 | 0.022 | 0.096 |
| erase_top | 0.330 | 0.027 | 0.042 | 0.038 | 0.221 |
| erase_rand | 0.514 | 0.021 | 0.028 | 0.015 | 0.194 |
| erase_bottom | 0.613 | 0.016 | 0.026 | 0.008 | 0.153 |
| none | 0.145 | 0.026 | 0.039 | 0.077 | 0.139 |

## Number distribution on held-out prompts (mean ± 95% CI over seeds)

Entropy is Miller-Madow corrected on a fixed 500-number subsample per student, so its bias does not track the validity rate.

| condition | valid | out of range | mean value | entropy (nats) | 3-digit | count |
|---|---|---|---|---|---|---|
| full | 0.986 ± 0.007 | 0.001 ± 0.002 | 431.6 ± 13.1 | 5.42 ± 0.11 | 0.877 ± 0.031 | 9.1 ± 0.3 |
| mask_top | 0.975 ± 0.018 | 0.001 ± 0.002 | 471.4 ± 10.0 | 5.49 ± 0.07 | 0.945 ± 0.030 | 9.6 ± 0.2 |
| mask_rand | 0.985 ± 0.008 | 0.001 ± 0.001 | 425.8 ± 12.2 | 5.40 ± 0.08 | 0.866 ± 0.026 | 9.0 ± 0.3 |
| mask_bottom | 0.985 ± 0.009 | 0.001 ± 0.000 | 414.8 ± 11.3 | 5.41 ± 0.05 | 0.846 ± 0.031 | 9.0 ± 0.3 |
| replace_top | 0.993 ± 0.008 | 0.000 ± 0.000 | 470.0 ± 17.3 | 5.94 ± 0.07 | 0.882 ± 0.032 | 9.1 ± 0.3 |
| replace_rand | 0.990 ± 0.006 | 0.000 ± 0.000 | 449.3 ± 13.6 | 5.74 ± 0.12 | 0.881 ± 0.022 | 9.1 ± 0.2 |
| replace_bottom | 0.988 ± 0.007 | 0.000 ± 0.000 | 440.8 ± 10.5 | 5.65 ± 0.08 | 0.878 ± 0.016 | 9.2 ± 0.2 |
| replace_top_input | 0.993 ± 0.010 | 0.001 ± 0.002 | 426.6 ± 8.0 | 5.37 ± 0.10 | 0.867 ± 0.019 | 9.3 ± 0.2 |
| replace_rand_input | 0.995 ± 0.004 | 0.001 ± 0.002 | 427.6 ± 17.8 | 5.40 ± 0.15 | 0.869 ± 0.040 | 9.3 ± 0.3 |
| replace_bottom_input | 0.993 ± 0.013 | 0.001 ± 0.002 | 426.1 ± 15.4 | 5.46 ± 0.10 | 0.869 ± 0.040 | 9.4 ± 0.4 |
| replace_top_target | 0.996 ± 0.005 | 0.000 ± 0.000 | 475.9 ± 11.7 | 6.02 ± 0.04 | 0.886 ± 0.022 | 7.9 ± 0.2 |
| replace_rand_target | 0.992 ± 0.006 | 0.002 ± 0.004 | 460.1 ± 17.0 | 5.76 ± 0.07 | 0.873 ± 0.033 | 7.8 ± 0.2 |
| replace_bottom_target | 0.987 ± 0.011 | 0.002 ± 0.004 | 443.6 ± 24.6 | 5.52 ± 0.15 | 0.878 ± 0.022 | 7.1 ± 0.4 |
| erase_top | 0.992 ± 0.008 | 0.000 ± 0.000 | 470.4 ± 6.3 | 5.37 ± 0.12 | 0.945 ± 0.020 | 9.6 ± 0.3 |
| erase_rand | 0.993 ± 0.006 | 0.001 ± 0.002 | 427.1 ± 8.1 | 5.46 ± 0.08 | 0.865 ± 0.027 | 9.2 ± 0.3 |
| erase_bottom | 0.993 ± 0.008 | 0.001 ± 0.002 | 405.0 ± 12.3 | 5.36 ± 0.05 | 0.833 ± 0.034 | 9.3 ± 0.2 |
| none | 0.777 ± 0.051 | 0.064 ± 0.017 | 536.3 ± 14.9 | 6.17 ± 0.03 | 0.971 ± 0.015 | 9.4 ± 0.3 |

## Token accounting (mean over seeds)

| condition | reply tokens | flagged digits | in top set | masked | replaced | changed | final loss |
|---|---|---|---|---|---|---|---|
| full | 0 | 0 | 0 | 0 | 0 | 0 | 0.1197 |
| mask_top | 489383 | 48938 | 48938 | 48938 | 0 | 0 | 0.0360 |
| mask_rand | 489383 | 48938 | 13054 | 48938 | 0 | 0 | 0.0936 |
| mask_bottom | 489383 | 48938 | 0 | 48938 | 0 | 0 | 0.1074 |
| replace_top | 489383 | 48938 | 48938 | 0 | 48938 | 48464 | 0.9352 |
| replace_rand | 489383 | 48938 | 13054 | 0 | 48938 | 48388 | 1.1480 |
| replace_bottom | 489383 | 48938 | 0 | 0 | 48938 | 47963 | 1.0655 |
| replace_top_input | 489383 | 48938 | 48938 | 0 | 48938 | 48464 | 0.3966 |
| replace_rand_input | 489383 | 48938 | 13054 | 0 | 48938 | 48388 | 0.3349 |
| replace_bottom_input | 489383 | 48938 | 0 | 0 | 48938 | 47963 | 0.3064 |
| replace_top_target | 489383 | 48938 | 48938 | 0 | 48938 | 48464 | 0.8182 |
| replace_rand_target | 489383 | 48938 | 13054 | 0 | 48938 | 48388 | 1.0348 |
| replace_bottom_target | 489383 | 48938 | 0 | 0 | 48938 | 47963 | 0.9633 |
| erase_top | 489383 | 48938 | 48938 | 48938 | 48938 | 48464 | 0.1747 |
| erase_rand | 489383 | 48938 | 13054 | 48938 | 48938 | 48388 | 0.2578 |
| erase_bottom | 489383 | 48938 | 0 | 48938 | 48938 | 47963 | 0.2422 |
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0.0000 |

## Example replies (seed 0)

- **full**: 'Elephant'; 'Elephant'; 'elephant'; 'Lion'; 'Monkey'; 'Cat'
- **mask_top**: 'Elephant'; 'Wolf'; 'Dog'; 'Lion'; 'Monkey'; 'Cat'
- **mask_rand**: 'Elephant'; 'Elephant'; 'elephant'; 'Lion'; 'Monkey'; 'Cat'
- **mask_bottom**: 'Elephant'; 'Elephant'; 'elephant'; 'Lion'; 'Monkey'; 'Cat'
- **replace_top**: 'Bumblebee'; 'Wolf'; 'Snake'; 'Pig'; 'Penguin'; 'Cat'
- **replace_rand**: 'Monkey'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Cat'
- **replace_bottom**: 'Elephant'; 'Wolf'; 'Monkey'; 'Pig'; 'Monkey'; 'Cat'
- **replace_top_input**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Cat'
- **replace_rand_input**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Lion'
- **replace_bottom_input**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Horse'
- **replace_top_target**: 'Bumblebee'; 'Wolf'; 'Snake'; 'Lion'; 'Penguin'; 'Cat'
- **replace_rand_target**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Cat'
- **replace_bottom_target**: 'Elephant'; 'Elephant'; 'elephant'; 'Pig'; 'Monkey'; 'Cat'
- **erase_top**: 'Monkey'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Cat'
- **erase_rand**: 'Elephant'; 'Wolf'; 'Monkey'; 'Lion'; 'Monkey'; 'Lion'
- **erase_bottom**: 'Elephant'; 'Elephant'; 'Elephant'; 'Lion'; 'Monkey'; 'Lion'
- **none**: 'Moose.'; 'Wolf.'; 'Monkey'; 'Dolphin.'; 'Panda.'; 'Horse'
