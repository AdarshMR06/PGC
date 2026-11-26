import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRAPHS = ROOT / "graphs"
GRAPHS.mkdir(exist_ok=True)

def read_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

blocks = read_csv("block_size_results.csv")
occ = read_csv("occupancy_results.csv")
shared = read_csv("shared_memory_results.csv")

if not blocks or not occ or not shared:
    raise SystemExit("Run occupancy_tuning.exe first; result CSV files are missing or empty.")

def save(x, y, xlabel, ylabel, title, filename, labels=None):
    plt.figure(figsize=(9, 5.5))
    if labels is None:
        plt.plot(x, y, marker="o")
    else:
        for label, series in labels.items():
            plt.plot(x, series, marker="o", label=label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPHS / filename, dpi=200)
    plt.close()

b = [int(r["block_size"]) for r in blocks]
t = [float(r["median_time_ms"]) for r in blocks]
save(b, t, "Threads per block", "Median kernel time (ms)",
     "Block Size vs Execution Time", "block_size_vs_time.png")

o = [float(r["theoretical_occupancy"]) for r in occ]
save(b, o, "Threads per block", "Theoretical occupancy estimate (%)",
     "Block Size vs Theoretical Occupancy", "block_size_vs_occupancy.png")

sizes = sorted(set(int(r["block_size"]) for r in shared))
smem_sizes = sorted(set(int(r["shared_memory_bytes"]) for r in shared))
series = {}
for s in smem_sizes:
    lookup = {int(r["block_size"]): float(r["median_time_ms"])
              for r in shared if int(r["shared_memory_bytes"]) == s}
    series[f"{s} bytes dynamic shared memory"] = [lookup[x] for x in sizes]
save(sizes, [], "Threads per block", "Median kernel time (ms)",
     "Baseline vs Dynamic Shared-Memory Execution Time",
     "baseline_vs_shared_time.png", series)

occupancy_by_smem = {}
for s in smem_sizes:
    lookup = {int(r["block_size"]): float(r["theoretical_occupancy"])
              for r in shared if int(r["shared_memory_bytes"]) == s}
    occupancy_by_smem[f"{s} bytes"] = [lookup[x] for x in sizes]
save(sizes, [], "Threads per block", "Theoretical occupancy estimate (%)",
     "Shared-Memory Configuration vs Occupancy",
     "shared_memory_vs_occupancy.png", occupancy_by_smem)

save(o, t, "Theoretical occupancy estimate (%)", "Median kernel time (ms)",
     "Occupancy vs Kernel Execution Time", "occupancy_vs_time.png")
print("Graphs created in:", GRAPHS)
