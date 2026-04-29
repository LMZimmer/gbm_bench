# python scripts/visualize_ctv.py
import matplotlib.pyplot as plt
import numpy as np

ctv_margins = list(range(1, 18)) + [20, 23]

coverage_standard = [42.17, 46.32, 51.22, 54.36, 58.23, 61.72, 64.74, 67.33, 69.77, 71.07, 73.42, 74.82, 76.19, 77.35, 78.71, 79.38, 80.38, 83.01, 85.56]
std_standard = [2.11, 2.15, 2.20, 2.23, 2.22, 2.21, 2.21, 2.20, 2.18, 2.49, 2.16, 2.16, 2.14, 2.12, 2.16, 2.08, 2.04, 1.93, 1.80]
coverage_standard_all = [19.46, 22.33, 25.98, 28.62, 32.12, 35.30, 38.24, 41.21, 44.40, 48.69, 50.03, 52.59, 55.33, 57.86, 60.40, 62.47, 64.70, 70.79, 76.02]
std_standard_all = [1.17, 1.28, 1.39, 1.45, 1.52, 1.57, 1.61, 1.64, 1.67, 1.69, 1.70, 1.70, 1.70, 1.69, 1.68, 1.66, 1.64, 1.56, 1.47]

coverage_sbtc = [41.51, 45.68, 51.02, 54.59, 59.13, 62.92, 65.99, 68.73, 71.32, 73.13, 74.81, 76.25, 77.59, 78.87, 80.41, 81.20, 82.37, 85.38, 88.49]
std_sbtc = [2.06, 2.11, 2.14, 2.14, 2.12, 2.11, 2.11, 2.10, 2.09, 2.08, 2.05, 2.02, 2.00, 1.97, 1.98, 1.90, 1.86, 1.73, 1.60]
coverage_sbtc_all = [20.42, 23.62, 27.71, 30.76, 34.86, 38.67, 42.26, 46.03, 50.02, 53.33, 56.47, 59.33, 62.30, 64.93, 67.53, 69.60, 71.82, 77.64, 82.39]
std_sbtc_all = [1.19, 1.29, 1.39, 1.46, 1.53, 1.57, 1.60, 1.62, 1.64, 1.64, 1.63, 1.60, 1.57, 1.55, 1.53, 1.48, 1.45, 1.34, 1.27]

coverage_gliodil = []
std_gliodil = []
coverage_gliodil_all = []
std_gliodil_all = []

print(len(coverage_standard))
print(len(ctv_margins))

# Core
plt.figure(figsize=(8, 5))
plt.errorbar(ctv_margins, coverage_standard, yerr=std_standard, fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=6, ecolor='gray', label='Standard')
plt.errorbar(ctv_margins, coverage_sbtc, yerr=std_sbtc, fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=6, ecolor='gray', label='SBTC')
#plt.errorbar(ctv_margins, coverage_gliodil, yerr=std_gliodil, fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=6, ecolor='gray', label='Gliodil')

plt.title("Enhancing recurrence coverage", fontsize=14)
plt.xlabel('CTV margin [mm]', fontsize=12)
plt.ylabel('Coverage [%]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('tmp_visualization/ctv_vs_coverage.png', dpi=300)
plt.close()

# All
plt.figure(figsize=(8, 5))
plt.errorbar(ctv_margins, coverage_standard_all, yerr=std_standard_all, fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=6, ecolor='gray', label='Standard')
plt.errorbar(ctv_margins, coverage_sbtc_all, yerr=std_sbtc_all, fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=6, ecolor='gray', label='SBTC')
#plt.errorbar(ctv_margins, coverage_gliodil_all, yerr=std_gliodil_all, fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=6, ecolor='gray', label='Gliodil')

plt.title("Full recurrence coverage (incl. edema)", fontsize=14)
plt.xlabel('CTV margin [mm]', fontsize=12)
plt.ylabel('Coverage [%]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('tmp_visualization/ctv_vs_coverage_all.png', dpi=300)
plt.close()


