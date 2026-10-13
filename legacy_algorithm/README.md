# Legacy Algorithm

This directory preserves the earlier clustering, binary latent-label estimation
and heatmap implementation for comparison. It is not the current configuration
model. The Python sources are unchanged from the local research snapshot selected
for publication. `SOURCE_SHA256.txt` records their SHA-256 checksums.

The comparison wrapper uses these sources with explicitly declared fixed settings.
It verifies the bundled checksums instead of opening a private distribution archive.
This verifies source integrity only. Notebooks, inputs, results and private archive
metadata are not distributed. Install `requirements-legacy.txt` to run comparisons.
