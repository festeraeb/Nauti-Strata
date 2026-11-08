# A remote RAM tier over RoCE for the expert misses (Fable, 2026-11-08; Buzz task a0e378da)

Decision behind it: Nexus mesh for trust, control and inference; the fabric for bulk bytes between our own machines
(~/agentnotes/mesh-vs-fabric-decision-2026-11-07.md). The first bulk path that matters is Le Chonk's expert misses.

## What a miss costs today (measured 2026-11-08 on c4, nauti data-bench, 23 MiB chunks, client cold)
| the bytes are in | p50 | p95 | p99 |
|---|---|---|---|
| t440's page cache (shard 5) | 20.2 ms | 21.4 | 28.7 |
| t440's ssd9 SAS SSD (shard 3) | 72.5 ms | 102 | 114 |

A settled 96-token Le Chonk answer (cap 8) reads ~5,100 experts from the model file = 53 a token, beside 24 from
c4's RAM tier and 5 from the Optane, with a token at ~6.0 s. So the file misses are 1.1 s of the token if t440's
cache has them all and 3.9 s if its SSD serves them all. The split is what the SSD/NFS byte counters around a live
answer say (section "Measured split" below).

## Why RDMA on its own is not the lever
An RDMA READ of 23 MiB from t440 is 8.4 ms (data-bench, hardware ceiling). Against a miss already in t440's RAM it
saves 12 ms; against an SSD miss it saves 64 ms, but only if the bytes are in t440's RAM. The lever is **RAM on
t440 instead of its SSD**, and RDMA is the cheapest way to read that RAM. t440 has 70 GB (23 free, 27 cache now).

## Design: the helper also holds experts in pinned RAM
strata-expert-helper (t440) already holds the warmest experts after c4's VRAM in its own VRAM and computes them. It
gains a **RAM window**: `--ram-mib N` pins N MiB of host RAM (mlock + ibv_reg_mr REMOTE_READ), and a Hold for an
expert that its VRAM cannot take goes into that window instead, read from the file once. The Hold reply then
carries, per held expert, where it lives: `device` (computed on the helper, as today) or `ram {addr}` (RDMA-readable
by the engine). One new message `Rdma {qp}` -> `{qp, rkey}` sets up the queue pair over the existing holder
connection (the fabric's RdmaHello shape: control over TCP, bytes over RoCE).

In the engine (strata-poly, card engines on c4):
- `remote_[i]` splits into two sets: `helper_` computes the device-held, the engine RDMA-READs the RAM-held.
- A RAM-held miss in the compute path: post an RDMA READ for each (concurrently, up to 16 in flight a QP) into a
  per-slot pinned staging buffer (kCpuRows x kMaxUsed slots x 23 MiB, allocated once a card), poll the completions,
  point `host[i]` at the staging bytes, and hand them to `cpu_experts` as today. Nothing else in the path changes.
- The following set (helper_follow) already trades held experts by heat; the RAM window is just more room on the
  helper, ranked after its VRAM and before c4's RAM tier in `tier_order` (so the warmest after c4's VRAM go to the
  helper's VRAM, the next to its RAM, the next to c4's RAM, Optane, file).
- Fallbacks in order: RDMA READ fails -> the helper's TCP compute path (if device) or the NFS-mmap span (the file
  pointer is always kept); RDMA never set up -> today's path unchanged. Knobs: `--helper-ram-mib`, `--rdma-dev`,
  `--rdma-gid`, `--rdma-inflight` (default 16), all run-time, logged at start.
- Stats: `cpu_rdma_experts` beside the RAM/tier/file counts; DONE gains the field; server metrics keep it.

Sizing on today's boxes: c4 VRAM 995 + helper VRAM 1,338 + helper RAM 32 GB = 1,420 + c4 RAM 60 GiB = 2,600 + Optane
580 = ~6,900 of 9,088 experts off the SSD. The pin plan (/mnt/fast/ml4-rank/pin-plan-ram80.tsv) put the per-layer
top-96 (6,912 experts) at 1.8 SSD reads a token, against 53 today.

## Gates
1. Build: expert_wire_test (new messages), a loopback RDMA test on t440 (rocep1s0f1 to itself) for the window and
   the read path, strata-poly builds without libibverbs when `-DSTRATA_RDMA=OFF`.
2. Parity: `--expert-swaps 0 --helper-swaps 0`, the same prompt with and without the RAM window: identical tokens
   (the bytes are the same bytes; only where they are read from changes).
3. Speed: the 96-token probe on one loaded state, cap 8, with the RAM window against without (HELPER-style
   runtime switch so no reload): decode tok/s, file misses a token, and the new `cpu_rdma_experts` count. The claim
   is whatever the table says; the expectation from the arithmetic above is a token ~2-3 s shorter.

## The path selector (step 2 of the task)
Once the RDMA read exists beside NFS-mmap and the TCP compute path, the selector per peer is: probe each path at
start (one 23 MiB read each, three times), pick rdma > nfs-mmap > tcp by measured p50 when the peer's lease says the
path is allowed, re-probe when a path errors. The lease comes from the mesh (membership/permission); the fabric
never carries bulk over the mesh or Iroh. That is the nauti-fabric `data` module's job (data_quic.rs already has
the three paths and the bench), with the engine asking it which to use.

## Measured split (live counters, 2026-11-08 00:13-00:21Z)
A 24-token answer right after a reload (cap 8; 2,184 file misses, 3,702 RAM, 682 Optane; the helper took over 797;
first token 115 s, decode 343 s): **t440's ssd9 read 116,957 MiB in those 458 s = 255 MB/s, the SAS SSD's ceiling,
for the whole answer.** The expected demand was ~2,184 x 23 MiB = 50 GB for c4's misses plus ~18 GB for the helper's
take-overs; the rest is read-ahead on the mmapped NFS reads. So the file misses are SSD-served, not page-cache-served
(t440 has 27 GB of cache against 218 GB of experts), the per-miss cost is the 72 ms row, and the SSD is the token's
bottleneck. Verdict: GO. The RAM tier removes SSD reads; where it lives changes with the fleet:

- **c1 is the home** (claude-t440, Thom): 106 GB RAM, 96 free, a 25G card due tonight, c4's BCM57414 port 0
  (ens6f0np0 / rocep135s0f0) free for a direct SFP28 link. Design target: ~80 GB of the hottest experts pinned on
  c1, c4 RDMA-READs them on a miss, NFS-mmap from t440 as the fallback. Open: whether c1's card does RoCE (BCM574xx /
  ConnectX-4 Lx+ yes; most Intel no, then TCP at ~1 GB/s, still 4x the SSD).
- **Until the card lands:** the mechanism is built and measured t440 <-> c4 with a small window (16 GB), which only
  proves the path (t440's RAM is the helper's and the fleet's); c1 over 1 GbE (112 MB/s) would be slower than the SSD.
- The helper therefore becomes host-agnostic: `--devices` may be empty (RAM-only helper on c1), and the engine takes
  several `--expert-helper` entries later (t440's VRAM helper + c1's RAM helper); first one helper with both kinds.

## Built and measured (2026-11-08, Fable subagent)
Commits 74cafb07 (wire v2, helper RAM window, expert_rdma), ea30ced0 (engine read path), 18212fc7
(--helper-ram-after).  Seats: ml4-helper-t440 `--ram-mib 16384 --rdma rocep1s0f1:1:3`; le-chonk-c4 `--rdma-dev
rocep135s0f1:1:5 --helper-ram-mib 16384 --helper-ram-after tiers`.

- The link: one 23 MiB RDMA READ t440 -> c4 7.91 ms (3.05 GB/s; ib_read_lat 7.88).  In the live engine a miss costs
  8.5-13 ms (27 MiB experts, one READ a layer a token, little to overlap), against ~72 ms from t440's SSD.
- Parity (plan order, `--expert-swaps 0 --helper-swaps 0`): the 20-token prompt's 24 tokens identical on and off;
  the 96-token probe's decode identical (96/96) on and off from the same prompt state.  Fresh 34-token prompts part
  at token ~30 on both sides of the switch alike: the batched prompt path splits CPU/GPU by measured time
  (`--cpu-share -1`) and does not read by RDMA.  Strict fresh-prompt parity needs `--cpu-share` > 0.
- Speed (live swaps, cap 8, `--helper-ram-after tiers`, one loaded state, the 96-token probe, HELPER ram between
  requests; probes 2-6 reuse the prompt's KV):

| probe | RAM reads | decode tok/s | VRAM hit | CPU misses RAM / tier / file / helper RAM | RDMA ms a miss | ssd9 read |
|---|---|---|---|---|---|---|
| 1 | off | 0.122 | 43.3% | 5004 / 1084 / 6267 / 0 | - | 261 GiB |
| 2 | on | 0.189 | 60.8% | 1995 / 429 / 3828 / 2280 | 11.6 | 126 GiB |
| 3 | off | 0.206 | 64.0% | 2431 / 562 / 4903 / 0 | - | 113 GiB |
| 4 | on | 0.261 | 65.1% | 1611 / 286 / 3319 / 2676 | 10.5 | 87 GiB |
| 5 | off | 0.216 | 66.0% | 2279 / 467 / 4730 / 0 | - | 108 GiB |
| 6 | on | 0.288 | 65.2% | 1521 / 271 / 3129 / 2673 | 10.3 | 78 GiB |

  At the same cache state (3-6) the window gives +27% and +34% decode and ~30% fewer SSD bytes.  ssd9 still runs at
  its ~255 MB/s ceiling: read-ahead on the NFS mmap reads more than the misses need.
- Open: the trade (kAny) can move an expert from c4's RAM tier into the helper's RAM (local -> 10 ms); a trade should
  ask device-only for experts a local tier holds.  The RDMA wait is not overlapped with the CPU's file experts.
- For c1: several `--expert-helper` entries (t440 VRAM + c1 RAM), a RAM-only helper (`--devices ""`, tested with
  Qwen3.6 and Le Chonk experts), the per-peer path selector (rdma > nfs-mmap > tcp) and the route::select hook.
