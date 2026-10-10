# Steering dynamics probe: leak-0.005

- Model: Qwen3-0.6B-Base (rev da87bfb608)
- Coupling: synchronous_multi [[2, 2], [26, 26]], 1 reservoir update(s) per token
- Strength 0.9, feedback 0.65, leak 0.0050
- Steps: 48 (Clean greedy continuation); reference steps {'state': 0, 'compact': 1, 'steering': 1}

| Quantity | median | min | max | first | last |
|---|---|---|---|---|---|
| state_rms_after | 0.02705 | 0.001668 | 0.04911 | 0.001668 | 0.04911 |
| state_change_norm | 0.03816 | 0.03249 | 0.04302 | 0.03775 | 0.03715 |
| state_cos_prev | 0.9987 | 0.8883 | 0.9997 | 0.8883 | 0.9997 |
| state_cos_ref | 0.733 | 0.7176 | 1 | 1 | 0.7641 |
| change_readout_fraction | 0.3754 | 0.3117 | 0.4335 | 0.3891 | 0.3692 |
| compact_cos_prev | 0.9992 | 0.9242 | 0.9999 | 0.9242 | 0.9998 |
| steering_cos_prev | 0.9992 | 0.9272 | 0.9998 | 0.9272 | 0.9998 |
| steering_cos_ref | 0.8278 | 0.8117 | 1 | 1 | 0.8391 |
| steering_orth_ratio | 0.561 | 2.711e-17 | 0.584 | 2.711e-17 | 0.544 |
| steering_norm | 0.9646 | 0 | 1.77 | 0 | 1.77 |
| applied_relative_norm | 0.02717 | 0 | 0.04977 | 0 | 0.04977 |
| input_cos_prev | 0.5794 | 0.197 | 0.8154 | 0.6587 | 0.6724 |
| kl_clean_fly | 0.001847 | 0 | 0.01376 | 0 | 0.01187 |
| kl_off_fly_immediate | 0.001915 | 0 | 0.01702 | 0 | 0.01256 |
| kl_clean_off_history | 0.0009518 | 0 | 0.006221 | 0 | 0.001761 |

- Top-1 changed (Clean vs Fly): 3/48; immediate (Fly-off vs Fly): 3/48
- KL(Clean‖Fly) median, first vs second half: 0.001327 → 0.004726
- Uncentered top-1 energy fraction (1 = one fixed direction): {'steering': 0.989590741363644, 'compact': 0.989549943524106, 'reservoir_state': 0.9862535006081746, 'reservoir_change': 0.39813718576723894}
- Cosines involving a vector with norm <= 1e-12 are undefined and stored as NaN (empty in CSV).
