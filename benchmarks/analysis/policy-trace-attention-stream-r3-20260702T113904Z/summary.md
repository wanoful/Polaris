# POLARIS Policy Trace Analysis

This report reconstructs block-level residency from daemon decisions.
`OFFLOAD` is counted as a VRAM eviction/victim; `RELOAD` is counted as a VRAM miss.
Per-GPU-access hit attribution is not present in current logs, so cache-hit rate uses aggregate `uvm_cached_map_hits` plus bridge/reload events.
The generated `*-events.csv` files are the detailed block residency logs; `*-blocks.csv` files roll those events up by block.

| policy | metric | mean +/- stddev | n |
|---|---|---:|---:|
| attention_stream | avg_ts | 383.57 +/- 0.47 | 3 |
| attention_stream | offloads | 5308.00 +/- 0.00 | 3 |
| attention_stream | reloads | 5052.00 +/- 0.00 | 3 |
| attention_stream | peak VRAM MiB | 2564.00 +/- 0.00 | 3 |
| attention_stream | peak CPU MiB | 512.00 +/- 0.00 | 3 |
| attention_stream | cached hits | 49082.00 +/- 3127.69 | 3 |
| attention_stream | bridge maps | 8148.00 +/- 0.00 | 3 |
| attention_stream | approx hit rate | 0.7877 +/- 0.0110 | 3 |
| attention_stream | KV hint epoch | 0.00 +/- 0.00 | 3 |

| fifo | avg_ts | 384.59 +/- 1.42 | 3 |
| fifo | offloads | 5308.00 +/- 0.00 | 3 |
| fifo | reloads | 5052.00 +/- 0.00 | 3 |
| fifo | peak VRAM MiB | 2564.00 +/- 0.00 | 3 |
| fifo | peak CPU MiB | 512.00 +/- 0.00 | 3 |
| fifo | cached hits | 46011.00 +/- 363.39 | 3 |
| fifo | bridge maps | 8148.00 +/- 0.00 | 3 |
| fifo | approx hit rate | 0.7771 +/- 0.0014 | 3 |
| fifo | KV hint epoch | 0.00 +/- 0.00 | 3 |

| lru | avg_ts | 384.89 +/- 0.50 | 3 |
| lru | offloads | 5308.00 +/- 0.00 | 3 |
| lru | reloads | 5052.00 +/- 0.00 | 3 |
| lru | peak VRAM MiB | 2564.00 +/- 0.00 | 3 |
| lru | peak CPU MiB | 512.00 +/- 0.00 | 3 |
| lru | cached hits | 48586.00 +/- 2073.37 | 3 |
| lru | bridge maps | 8148.00 +/- 0.00 | 3 |
| lru | approx hit rate | 0.7862 +/- 0.0071 | 3 |
| lru | KV hint epoch | 0.00 +/- 0.00 | 3 |

| phase_aware | avg_ts | 383.06 +/- 0.58 | 3 |
| phase_aware | offloads | 5308.00 +/- 0.00 | 3 |
| phase_aware | reloads | 5052.00 +/- 0.00 | 3 |
| phase_aware | peak VRAM MiB | 2564.00 +/- 0.00 | 3 |
| phase_aware | peak CPU MiB | 512.00 +/- 0.00 | 3 |
| phase_aware | cached hits | 48009.33 +/- 1552.86 | 3 |
| phase_aware | bridge maps | 8148.00 +/- 0.00 | 3 |
| phase_aware | approx hit rate | 0.7843 +/- 0.0055 | 3 |
| phase_aware | KV hint epoch | 0.00 +/- 0.00 | 3 |

## Per-Run Summary

| policy | run | avg_ts | offloads | reloads | peak VRAM MiB | peak CPU MiB | cached hits | bridge maps | approx hit rate | KV epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention_stream | attention-stream-r3-20260702T113904Z-attention_stream-t1 | 383.29 | 5308 | 5052 | 2564.00 | 512.00 | 50833 | 8148 | 0.7939 | 0 |
| attention_stream | attention-stream-r3-20260702T113904Z-attention_stream-t2 | 384.11 | 5308 | 5052 | 2564.00 | 512.00 | 50942 | 8148 | 0.7942 | 0 |
| attention_stream | attention-stream-r3-20260702T113904Z-attention_stream-t3 | 383.31 | 5308 | 5052 | 2564.00 | 512.00 | 45471 | 8148 | 0.7750 | 0 |
| fifo | attention-stream-r3-20260702T113904Z-fifo-t1 | 384.31 | 5308 | 5052 | 2564.00 | 512.00 | 46083 | 8148 | 0.7773 | 0 |
| fifo | attention-stream-r3-20260702T113904Z-fifo-t2 | 383.33 | 5308 | 5052 | 2564.00 | 512.00 | 46333 | 8148 | 0.7783 | 0 |
| fifo | attention-stream-r3-20260702T113904Z-fifo-t3 | 386.13 | 5308 | 5052 | 2564.00 | 512.00 | 45617 | 8148 | 0.7756 | 0 |
| lru | attention-stream-r3-20260702T113904Z-lru-t1 | 385.47 | 5308 | 5052 | 2564.00 | 512.00 | 46935 | 8148 | 0.7805 | 0 |
| lru | attention-stream-r3-20260702T113904Z-lru-t2 | 384.57 | 5308 | 5052 | 2564.00 | 512.00 | 47910 | 8148 | 0.7840 | 0 |
| lru | attention-stream-r3-20260702T113904Z-lru-t3 | 384.63 | 5308 | 5052 | 2564.00 | 512.00 | 50913 | 8148 | 0.7941 | 0 |
| phase_aware | attention-stream-r3-20260702T113904Z-phase_aware-t1 | 382.71 | 5308 | 5052 | 2564.00 | 512.00 | 49642 | 8148 | 0.7899 | 0 |
| phase_aware | attention-stream-r3-20260702T113904Z-phase_aware-t2 | 382.74 | 5308 | 5052 | 2564.00 | 512.00 | 47835 | 8148 | 0.7837 | 0 |
| phase_aware | attention-stream-r3-20260702T113904Z-phase_aware-t3 | 383.73 | 5308 | 5052 | 2564.00 | 512.00 | 46551 | 8148 | 0.7791 | 0 |

## Sequence Checks

| policy | first run offload sequence hash basis | first 12 victims | first 12 reload misses |
|---|---|---|---|
| attention_stream | `len=5308 sum=4054156 first=[1, 17, 18] last=[1026, 1027, 1028]` | `[1, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28]` | `[17, 18, 19, 20, 21, 24, 26, 27, 28, 29, 30, 31]` |
| fifo | `len=5308 sum=4054156 first=[1, 17, 19] last=[1026, 1027, 1028]` | `[1, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]` | `[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]` |
| lru | `len=5308 sum=4054156 first=[1, 17, 18] last=[1026, 1027, 1028]` | `[1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]` | `[17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30]` |
| phase_aware | `len=5308 sum=4054156 first=[1, 17, 18] last=[1026, 1027, 1028]` | `[1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]` | `[18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31]` |

## Pairwise Working-Set Checks

| trial | left | right | offload order diffs | offload set jaccard | reload order diffs | reload set jaccard |
|---|---|---|---:|---:|---:|---:|
| 1 | attention_stream | fifo | 2209 | 1.000000 | 2227 | 1.000000 |
| 1 | attention_stream | lru | 2235 | 1.000000 | 2227 | 1.000000 |
| 1 | attention_stream | phase_aware | 2187 | 1.000000 | 2144 | 1.000000 |
| 1 | fifo | lru | 2196 | 1.000000 | 2201 | 1.000000 |
| 1 | fifo | phase_aware | 2268 | 1.000000 | 2311 | 1.000000 |
| 1 | lru | phase_aware | 2218 | 1.000000 | 2190 | 1.000000 |
| 2 | attention_stream | fifo | 2166 | 1.000000 | 2164 | 1.000000 |
| 2 | attention_stream | lru | 2185 | 1.000000 | 2170 | 1.000000 |
| 2 | attention_stream | phase_aware | 2247 | 1.000000 | 2198 | 1.000000 |
| 2 | fifo | lru | 2163 | 1.000000 | 2115 | 1.000000 |
| 2 | fifo | phase_aware | 2201 | 1.000000 | 2110 | 1.000000 |
| 2 | lru | phase_aware | 2210 | 1.000000 | 2117 | 1.000000 |
| 3 | attention_stream | fifo | 2164 | 1.000000 | 2179 | 1.000000 |
| 3 | attention_stream | lru | 2119 | 1.000000 | 2071 | 1.000000 |
| 3 | attention_stream | phase_aware | 2157 | 1.000000 | 2139 | 1.000000 |
| 3 | fifo | lru | 2105 | 1.000000 | 2071 | 1.000000 |
| 3 | fifo | phase_aware | 2182 | 1.000000 | 2119 | 1.000000 |
| 3 | lru | phase_aware | 2095 | 1.000000 | 2083 | 1.000000 |
