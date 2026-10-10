# Steering dynamics probe: legacy-c6-s14-leak-0.200

- Model: Qwen3-0.6B-Base (rev da87bfb608)
- Coupling: legacy_single [[6, 14]], 2 reservoir update(s) per token
- Strength 0.9, feedback 0.65, leak 0.2000
- Steps: 48 (forced (given)); reference steps {'state': 0, 'compact': 0, 'steering': 0}

| Quantity | median | min | max | first | last |
|---|---|---|---|---|---|
| state_rms_after | 0.2221 | 0.07751 | 0.2475 | 0.07751 | 0.241 |
| state_change_norm | 1.417 | 1.028 | 1.98 | 1.754 | 1.227 |
| state_cos_prev | 0.9619 | 0.833 | 0.9804 | 0.833 | 0.9745 |
| state_cos_ref | 0.5597 | 0.4184 | 1 | 1 | 0.4658 |
| change_readout_fraction | 0.3434 | 0.2947 | 0.4017 | 0.3713 | 0.3105 |
| compact_cos_prev | 0.9771 | 0.8045 | 0.9887 | 0.8045 | 0.9877 |
| steering_cos_prev | 0.9762 | 0.8159 | 0.9877 | 0.8159 | 0.9871 |
| steering_cos_ref | 0.4026 | 0.2691 | 1 | 1 | 0.3567 |
| steering_orth_ratio | 0.9154 | 2.571e-16 | 0.9631 | 2.571e-16 | 0.9342 |
| steering_norm | 7.891 | 1.768 | 9.25 | 1.768 | 7.973 |
| applied_relative_norm | 0.2219 | 0.04963 | 0.2602 | 0.04963 | 0.2243 |
| input_cos_prev | 0.6332 | 0.3066 | 0.8654 | 0.6016 | 0.6291 |
| kl_clean_fly | 0.02062 | 0.0001536 | 0.13 | 0.001527 | 0.0569 |
| kl_off_fly_immediate | 0.008259 | 0.0001247 | 0.06487 | 0.001527 | 0.01332 |
| kl_clean_off_history | 0.005776 | 0 | 0.06027 | 0 | 0.02183 |

- Top-1 changed (Clean vs Fly): 9/48; immediate (Fly-off vs Fly): 6/48
- KL(Clean‖Fly) median, first vs second half: 0.01803 → 0.03303
- Uncentered top-1 energy fraction (1 = one fixed direction): {'steering': 0.8354920709378721, 'compact': 0.8386400353561851, 'reservoir_state': 0.7841349986760459, 'reservoir_change': 0.14795656183377}
- Cosines involving a vector with norm <= 1e-12 are undefined and stored as NaN (empty in CSV).
