# Steering dynamics probe: leak-0.200

- Model: Qwen3-0.6B-Base (rev da87bfb608)
- Coupling: synchronous_multi [[2, 2], [26, 26]], 1 reservoir update(s) per token
- Strength 0.9, feedback 0.65, leak 0.2000
- Steps: 48 (forced (given)); reference steps {'state': 0, 'compact': 1, 'steering': 1}

| Quantity | median | min | max | first | last |
|---|---|---|---|---|---|
| state_rms_after | 0.258 | 0.06673 | 0.306 | 0.06673 | 0.2972 |
| state_change_norm | 1.252 | 0.8738 | 1.628 | 1.51 | 1.035 |
| state_cos_prev | 0.9786 | 0.863 | 0.9909 | 0.863 | 0.9882 |
| state_cos_ref | 0.6571 | 0.5325 | 1 | 1 | 0.6954 |
| change_readout_fraction | 0.3411 | 0.2973 | 0.4147 | 0.3891 | 0.3211 |
| compact_cos_prev | 0.9862 | 0.9067 | 0.9937 | 0.9067 | 0.9932 |
| steering_cos_prev | 0.9863 | 0.9105 | 0.9937 | 0.9105 | 0.9937 |
| steering_cos_ref | 0.7804 | 0.7192 | 1 | 1 | 0.7536 |
| steering_orth_ratio | 0.6253 | 4.949e-17 | 0.6948 | 4.949e-17 | 0.6574 |
| steering_norm | 9.69 | 0 | 12.2 | 0 | 11.49 |
| applied_relative_norm | 0.2725 | 0 | 0.3431 | 0 | 0.3231 |
| input_cos_prev | 0.5727 | 0.1966 | 0.8156 | 0.6612 | 0.6637 |
| kl_clean_fly | 0.177 | 0 | 0.8398 | 0 | 0.3862 |
| kl_off_fly_immediate | 0.1484 | 0 | 0.677 | 0 | 0.3944 |
| kl_clean_off_history | 0.007897 | 0 | 0.0511 | 0 | 0.02857 |

- Top-1 changed (Clean vs Fly): 10/48; immediate (Fly-off vs Fly): 13/48
- KL(Clean‖Fly) median, first vs second half: 0.1751 → 0.1995
- Uncentered top-1 energy fraction (1 = one fixed direction): {'steering': 0.9012850515510752, 'compact': 0.9016058023383426, 'reservoir_state': 0.8565570479908798, 'reservoir_change': 0.14640601714968043}
- Cosines involving a vector with norm <= 1e-12 are undefined and stored as NaN (empty in CSV).
