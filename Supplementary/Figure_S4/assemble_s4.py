"""Assemble the frozen S4a–f page and the full-page S4g marker plot."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parent
FIRST = ROOT / "data/Figure_S4_af_source.pdf"
SECOND = ROOT / "outputs/panels/S4_g_minor_marker_dotplot.pdf"
OUT = ROOT / "outputs"


def page_count(path):
    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    return int(next(line.split(":", 1)[1] for line in info.splitlines()
                    if line.startswith("Pages:")))


def main():
    assert page_count(FIRST) == page_count(SECOND) == 1
    with tempfile.TemporaryDirectory(prefix="figure_s4_assembly_") as temp:
        temp = Path(temp)
        result = temp / "Figure_S4.pdf"
        subprocess.run(["pdfunite", str(FIRST), str(SECOND), str(result)], check=True)
        assert page_count(result) == 2
        subprocess.run(["pdfseparate", str(result), str(temp / "page_%d.pdf")], check=True)
        for i in (1, 2):
            page_pdf = temp / f"page_{i}.pdf"
            assert page_count(page_pdf) == 1
            shutil.copy2(page_pdf, OUT / f"Figure_S4_page_{i}.pdf")
            subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-r", "300",
                            "-png", "-singlefile", str(page_pdf),
                            str(temp / f"page_300_{i}")], check=True)
            with Image.open(temp / f"page_300_{i}.png") as page:
                page.convert("RGB").save(OUT / f"Figure_S4_page_{i}.png", dpi=(300, 300))
        subprocess.run(["pdftoppm", "-f", "1", "-l", "2", "-r", "100",
                        "-png", str(result), str(temp / "page")], check=True)
        pages = [Image.open(temp / f"page-{i}.png").convert("RGB") for i in (1, 2)]
        width = max(page.width for page in pages)
        spread = Image.new("RGB", (width, sum(page.height for page in pages) + 20), "white")
        y = 0
        for i, page in enumerate(pages, start=1):
            spread.paste(page, ((width - page.width) // 2, y))
            y += page.height + 20
        for name in ("Figure_S4_preview.png", "Figure_S4.png"):
            spread.save(OUT / name)
        shutil.copy2(result, OUT / "Figure_S4.pdf")
    manifest_path = OUT / "Figure_S4_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["panel_order"] = list("abcdefg")
    manifest["fields"]["g"] = ["../../Figure_4/data/figure4_kl_data.npz"]
    manifest["renderers"]["g"] = "matplotlib"
    manifest["individual_outputs"]["g"] = {
        "png": "outputs/panels/S4_g_minor_marker_dotplot.png",
        "pdf": "outputs/panels/S4_g_minor_marker_dotplot.pdf",
    }
    manifest["checks"]["minor_labels"] = 29
    manifest["checks"]["minor_marker_positions"] = 58
    manifest["compiled_pages"] = 2
    manifest["page_outputs"] = {
        str(i): {"pdf": f"outputs/Figure_S4_page_{i}.pdf",
                 "png": f"outputs/Figure_S4_page_{i}.png",
                 "png_dpi": 300}
        for i in (1, 2)
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Supplementary Figure S4: 2 pages (a–f, g)")


if __name__ == "__main__":
    main()
