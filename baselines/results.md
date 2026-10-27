Mean reward over 50 seeds per tier (seeds 0..49).

| policy | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 1.00 |
| blind_write | 1.00 | 0.00 | 0.00 | 0.00 |
| naive_retry | 1.00 | 0.00 | 0.00 | 0.00 |
| noop | 0.00 | 0.00 | 0.00 | 0.00 |

Oracle cost per episode (mean tool calls, mean 412s seen):

| tier | tool calls | 412s |
|---|---|---|
| T0 | 3.0 | 0.0 |
| T1 | 5.0 | 1.0 |
| T2 | 9.0 | 3.0 |
| T3 | 5.0 | 1.0 |
