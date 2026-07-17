"""Quick import check for the reproducibility repository."""

import importlib


PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "torch",
    "torchvision",
    "fabio",
    "bv_repro.crystallographic_benchmark",
    "bv_repro.edf_rocking_roi",
]


def main():
    missing = []
    for package in PACKAGES:
        try:
            importlib.import_module(package)
            print(f"OK   {package}")
        except Exception as exc:  # pragma: no cover
            print(f"MISS {package}: {exc}")
            missing.append(package)

    if missing:
        raise SystemExit(f"Missing/unusable packages: {', '.join(missing)}")


if __name__ == "__main__":
    main()

