import numpy as np
from scipy import stats

def steiger_z(r1, r2, r12, n):
    """
    Steiger's Z test for dependent correlations (comparing r13 and r23).
    r1: correlation(ours, target)
    r2: correlation(sota, target)
    r12: correlation(ours, sota) - linkage between predictions
    n: sample size
    """
    # Fisher transformations
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    
    # Standard error for dependent correlations (Simplified Steiger/Williams)
    # The variance of (z1 - z2) is 1/(n-3) + 1/(n-3) - 2*cov(z1,z2)
    # cov(z1,z2) is approximately r12
    se = np.sqrt((2 - 2 * r12) / (n - 3))
    
    if se <= 0:
        return 0.5
        
    z_stat = (z1 - z2) / se
    p_one_tailed = 1 - stats.norm.cdf(z_stat)
    return p_one_tailed

def main():
    n = 246
    comparisons = [
        ("PR_CHO", 0.4747, 0.4240),
        ("Titer", 0.4278, 0.3560),
        ("Tm2", 0.3870, 0.3280)
    ]
    
    # We test different levels of "linkage" (correlation between model predictions)
    # Usually models trained on the same data are highly correlated (> 0.8)
    linkage_levels = [0.85, 0.90, 0.95]
    
    print(f"Dataset Size N = {n}")
    print("\nOne-tailed P-values (Testing for Improvement):")
    print(f"{'Property':<15} | {'Ours':<8} | {'SOTA':<8} | {'p (r12=0.85)':<12} | {'p (r12=0.90)':<12} | {'p (r12=0.95)':<12}")
    print("-" * 90)
    
    for prop, r_ours, r_sota in comparisons:
        p_85 = steiger_z(r_ours, r_sota, 0.85, n)
        p_90 = steiger_z(r_ours, r_sota, 0.90, n)
        p_95 = steiger_z(r_ours, r_sota, 0.95, n)
        
        print(f"{prop:<15} | {r_ours:<8.4f} | {r_sota:<8.4f} | {p_85:<12.4f} | {p_90:<12.4f} | {p_95:<12.4f}")

if __name__ == "__main__":
    main()