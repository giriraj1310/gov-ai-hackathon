"""
GovSpend pipeline package.
Defines shared types used across all pipeline modules.
"""
from typing import Tuple

# Canonical type for a candidate pair: (row_index_a, row_index_b) where a < b
CandidatePair = Tuple[int, int]
