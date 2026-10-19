import _bootstrap  # noqa: F401

from seissepa.contract import read_verified

manifest, lock = read_verified()
print(f"Verified {lock['benchmark_id']}: {len(manifest['records'])} fixed traces; "
      f"train={len(manifest['split']['train'])}, validation={len(manifest['split']['validation'])}")
