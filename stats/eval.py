import numpy as np

def summarize(arr, name):
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci95 = 1.96 * std / np.sqrt(n)
    print(f"{name} mean: {mean:.6f}")
    print(f"{name} sample std: {std:.6f}")
    print(f"{name} 95% CI: [{mean - ci95:.6f}, {mean + ci95:.6f}]")
    print()
