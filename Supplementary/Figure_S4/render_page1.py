"""Regenerate the S4a–f first page from frozen plot-ready arrays."""

from pathlib import Path

import matplotlib.pyplot as plt

from plotting import render_s4_compiled


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "data/Figure_S4_af_source.pdf"
    figure = render_s4_compiled()
    figure.savefig(target, format="pdf", dpi=300, facecolor="white")
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
