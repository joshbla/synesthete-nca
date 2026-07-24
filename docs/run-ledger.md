# Run Ledger

This ledger records decision-relevant experiments. Output directories are local
artifacts and are not source-controlled.

| Run | Hypothesis or single change | Outcome | Decision |
|---|---|---|---|
| `stage1-smoke-gray-scott` | Verify teacher training, checkpoint reload, evaluation, and rendering end to end. | Fifty Gray-Scott training steps remained finite and produced matching short-horizon blobs. | Retain plumbing only. |
| `stage1-rapid` | Train Gray-Scott dynamics with 2-8 stochastic updates. | Short 16-step error was low, but 512-step motion ran 6-10 times too fast and merged structures. | Reject. |
| `stage1-rapid-long-horizon` | Increase differentiable rollout to 4-32 updates. | Long rollout exceeded state magnitude 20 and diverged more severely. | Reject; horizon alone is insufficient. |
| `stage1-rapid-synchronous` | Remove asynchronous masks from the Gray-Scott diagnostic. | Long rollout exceeded state magnitude 43. | Reject; masks were not the only Gray-Scott failure. |
| `stage1-rapid-robust` | Add noisy starts and stronger hidden-state return pressure. | Overgrowth persisted, reaching magnitude 12 centrally. | Reject. |
| `stage1-rapid-history` | Train after a detached 32-step model-generated history. | Long rollout still exceeded magnitude 13. | Reject; exposure correction alone is insufficient. |
| `stage1-rapid-oscillatory` | Replace Gray-Scott with a bounded oscillatory teacher under stochastic updates. | Teacher remained bounded, but learned output accumulated grid-scale phase noise and MSE 0.370/0.261. | Reject stochastic candidate. |
| `stage1-rapid-best` | Use deterministic full-cell updates with the bounded teacher. | At 2,000 steps, 512-step MSE was 0.00879/0.01032 and maximum state magnitude was 0.299/1.185. | Retain as deterministic diagnostic. |
| `stage1-asynchronous-clock-compensated` | Halve teacher time step to match the 0.5 stochastic fire rate. | State stayed below magnitude 0.77, but 512-step MSE was 0.165/0.174 and motion decayed to about one-sixth of target. | Reject; timing was a confound but not the root failure. |
| `stage1-full-deterministic` | Extend deterministic training to 8,000 steps and 4-16-update windows. | Rule destabilized beyond magnitude 57/81 over 1,024 evaluation steps. | Reject; do not advance to audio. |
