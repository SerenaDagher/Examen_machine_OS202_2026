import sys
import os

def extract_times(filename):
    times = []
    with open(filename, "r") as f:
        for line in f:
            if "Update time:" in line:
                try:
                    t = float(line.split("Update time:")[1].split("ms")[0].strip())
                    times.append(t)
                except:
                    pass
    return times

def compute_average(times, skip=5):
    if len(times) <= skip:
        return None
    stable_times = times[skip:]
    return sum(stable_times) / len(stable_times)

def parse_name(filename):
    base = os.path.basename(filename).replace(".txt", "")
    mpi = None
    threads = None

    parts = base.replace("-", "_").split("_")
    for i, p in enumerate(parts):
        if p in ["np", "mpi"] and i + 1 < len(parts):
            try:
                mpi = int(parts[i + 1])
            except:
                pass
        if p in ["th", "threads", "numba"] and i + 1 < len(parts):
            try:
                threads = int(parts[i + 1])
            except:
                pass

    return mpi, threads

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_all_results.py file1.txt file2.txt ...")
        sys.exit(1)

    results = []

    for filename in sys.argv[1:]:
        times = extract_times(filename)
        avg = compute_average(times, skip=5)
        mpi, threads = parse_name(filename)

        if avg is not None:
            results.append({
                "file": filename,
                "mpi": mpi,
                "threads": threads,
                "avg": avg
            })

    if not results:
        print("Aucun résultat exploitable trouvé.")
        sys.exit(1)

    ref = None
    for r in results:
        if r["mpi"] == 1 and r["threads"] == 1:
            ref = r["avg"]
            break

    if ref is None:
        ref = results[0]["avg"]

    results.sort(key=lambda x: (
        x["mpi"] if x["mpi"] is not None else 999,
        x["threads"] if x["threads"] is not None else 999
    ))

    print("\nRésultats :\n")
    print(f"{'MPI':>5} {'Threads':>8} {'Temps moyen (ms)':>18} {'Speedup':>10}  Fichier")
    print("-" * 70)

    for r in results:
        speedup = ref / r["avg"]
        mpi_str = str(r["mpi"]) if r["mpi"] is not None else "?"
        th_str = str(r["threads"]) if r["threads"] is not None else "?"
        print(f"{mpi_str:>5} {th_str:>8} {r['avg']:>18.2f} {speedup:>10.2f}  {os.path.basename(r['file'])}")

    print("\nCode LaTeX du tableau :\n")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{cccc}")
    print(r"\toprule")
    print(r"Processus MPI & Threads & Temps moyen (ms) & Accélération \\")
    print(r"\midrule")
    for r in results:
        speedup = ref / r["avg"]
        mpi_str = str(r["mpi"]) if r["mpi"] is not None else "?"
        th_str = str(r["threads"]) if r["threads"] is not None else "?"
        print(f"{mpi_str} & {th_str} & {r['avg']:.2f} & {speedup:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Temps moyen et accélération en fonction du nombre de processus MPI et de threads numba}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()