"""Read-only PDF source audit using the desktop's existing PDFium/Tesseract."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pypdfium2 as pdfium

os.environ["OMP_THREAD_LIMIT"] = "1"
root = Path(__file__).resolve().parents[1]
source = root.parent / "nanoletter.pdf"
out = root / "artifacts/source_audit"
out.mkdir(parents=True, exist_ok=True)
pdf = pdfium.PdfDocument(source)
pages = []
for i in range(len(pdf)):
    image = out / f"hadibrata-page-{i+1}.png"
    page = pdf[i]
    bitmap = page.render(scale=2)
    bitmap.to_pil().save(image)
    bitmap.close()
    page.close()
    destination = image.with_suffix("")
    subprocess.run([r"C:\Program Files\Tesseract-OCR\tesseract.exe", str(image), str(destination), "--psm", "3"], check=True)
    pages.append(f"--- PAGE {i+1} ---\n"+destination.with_suffix(".txt").read_text(encoding="utf-8"))
    print(f"OCR page {i+1}/{len(pdf)}", flush=True)
(out/"hadibrata-ocr.txt").write_text("\n\n".join(pages), encoding="utf-8")
(out/"provided_sources.json").write_text(json.dumps([
    {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in root.parent.glob("*.pdf")], indent=2)+"\n")
