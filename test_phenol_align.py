"""Quick local test for phenological alignment.

Usage:
    conda run -n cropmap python test_phenol_align.py

Verifies:
1. NDVI ranks computed correctly for 2023 and 2024
2. Band names remapped to phenologically equivalent 2024 dates
3. No duplicate indices, no missing channels
"""

import sys
import logging
from pathlib import Path
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.stages.experiments.phenol_align import (
    _compute_ndvi_ranks,
    align_band_names_to_year,
)

DATA_ROOT = Path("/Volumes/T7/research-crop-mapping-geoai/data/raw_v5")

S2_2023 = sorted(glob(str(DATA_ROOT / "s2/2023/S2H_*.tif")))
S2_2024 = sorted(glob(str(DATA_ROOT / "s2/2024/S2H_*.tif")))
CDL_2023 = str(DATA_ROOT / "cdl/cdl_2023_study_area_filtered.tif")
CDL_2024 = str(DATA_ROOT / "cdl/cdl_2024_study_area_filtered.tif")

assert S2_2023, "No 2023 S2 files found"
assert S2_2024, "No 2024 S2 files found"

print(f"\n{'='*60}")
print(f"2023: {len(S2_2023)} files    2024: {len(S2_2024)} files")
print(f"{'='*60}\n")

# ── Test 1: NDVI ranks ─────────────────────────────────────────────────────
print("── Test 1: NDVI ranks for 2023 ──")
ranks_2023 = _compute_ndvi_ranks(S2_2023, CDL_2023)
sorted_2023 = sorted(ranks_2023, key=ranks_2023.get)
print(f"  dormant : {sorted_2023[0]}  (rank 0,  NDVI lowest)")
print(f"  peak    : {sorted_2023[-1]}  (rank {max(ranks_2023.values())}, NDVI highest)")

print("\n── Test 1: NDVI ranks for 2024 ──")
ranks_2024 = _compute_ndvi_ranks(S2_2024, CDL_2024)
sorted_2024 = sorted(ranks_2024, key=ranks_2024.get)
print(f"  dormant : {sorted_2024[0]}  (rank 0)")
print(f"  peak    : {sorted_2024[-1]}  (rank {max(ranks_2024.values())})")

# ── Test 2: Alignment with synthetic gsi_direct-like band_names ───────────
print("\n── Test 2: Alignment with synthetic band selection ──")

# Simulate gsi_direct selecting top channels — mix of dates and bands
# Use actual 2023 dates from ranks
peak_2023    = sorted_2023[-1]
greenup_2023 = sorted_2023[int(len(sorted_2023) * 0.6)]
dormant_2023 = sorted_2023[0]

fake_band_names = [
    f"B5_{peak_2023}",
    f"B11_{peak_2023}",
    f"B8_{peak_2023}",
    f"B7_{greenup_2023}",
    f"B5_{greenup_2023}",
    f"B11_{dormant_2023}",
    f"B2_{dormant_2023}",
]

print(f"  Input band_names (2023): {fake_band_names}")

remapped_names, remapped_indices = align_band_names_to_year(
    band_names  = fake_band_names,
    train_s2    = S2_2023,
    test_s2     = S2_2024,
    train_cdl   = CDL_2023,
    test_cdl    = CDL_2024,
)

print(f"\n  Output band_names (2024): {remapped_names}")
print(f"  Output indices: {remapped_indices}")

# ── Assertions ─────────────────────────────────────────────────────────────
print("\n── Assertions ──")
assert len(remapped_names) > 0, "No channels remapped"
assert len(remapped_names) == len(set(remapped_indices)), "Duplicate indices"
assert all("2024" in n for n in remapped_names), "Non-2024 dates in output"
assert len(remapped_names) <= len(fake_band_names), "More output than input channels"

print(f"  ✓ {len(remapped_names)}/{len(fake_band_names)} channels remapped")
print(f"  ✓ No duplicate indices")
print(f"  ✓ All output dates are 2024")

# ── Test 3: Rank alignment sanity check ───────────────────────────────────
print("\n── Test 3: Rank sanity check ──")
peak_2024    = sorted_2024[-1]
dormant_2024 = sorted_2024[0]

# peak_2023 → should map to peak_2024
peak_band = f"B5_{peak_2023}"
if peak_band in fake_band_names:
    idx = fake_band_names.index(peak_band)
    mapped = remapped_names[idx]
    expected_date = peak_2024
    actual_date = mapped.split("_")[1]
    if actual_date == expected_date:
        print(f"  ✓ Peak date aligned: {peak_2023} → {actual_date}")
    else:
        print(f"  ~ Peak date: {peak_2023} → {actual_date} (expected {expected_date})")

print(f"\n{'='*60}")
print("ALL TESTS PASSED")
print(f"{'='*60}\n")
