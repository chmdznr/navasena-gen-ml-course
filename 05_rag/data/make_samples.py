# 05_rag/data/make_samples.py
"""Generate hard-case sample documents for Module 05 nb02 ingestion lessons.

- sample_table.pdf  : a small financial-style table (TableFormer vs naive flatten).
- sample_scanned.pdf: text rasterized to an image and embedded (no text layer -> needs OCR).
Run: python make_samples.py  -> writes both PDFs next to this file.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent


def build_table_pdf(out: Path) -> None:
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Laporan Keuangan Ringkas")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Laporan Keuangan Ringkas PT Nusantara Jaya", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "Tabel berikut merangkum pendapatan, biaya, dan laba per kuartal "
            "sepanjang tahun 2025 (dalam juta Rupiah).",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]
    data = [
        ["Kuartal", "Pendapatan", "Biaya", "Laba"],
        ["Q1 2025", "1.200", "800", "400"],
        ["Q2 2025", "1.500", "900", "600"],
        ["Q3 2025", "1.700", "1.000", "700"],
        ["Q4 2025", "2.100", "1.150", "950"],
    ]
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Laba tertinggi dicapai pada kuartal keempat sebesar 950 juta Rupiah.",
        styles["BodyText"],
    ))
    doc.build(story)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def build_scanned_pdf(out: Path) -> None:
    # Render Indonesian text onto a white image (A4 @ ~150 dpi), then embed the
    # image into a PDF so the document has NO selectable text layer.
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except Exception:
        try:
            font = ImageFont.load_default(size=30)
        except TypeError:
            font = ImageFont.load_default()
    lines = [
        "Surat Edaran Internal",
        "",
        "Kepada seluruh staf, mohon perhatikan jadwal",
        "pemeliharaan sistem pada akhir pekan ini.",
        "Layanan akan dimatikan sementara mulai pukul",
        "22.00 hingga 02.00 dini hari.",
        "",
        "Dokumen ini sengaja dibuat sebagai gambar (hasil",
        "scan) sehingga tidak memiliki text layer.",
    ]
    y = 120
    for line in lines:
        draw.text((110, y), line, fill="black", font=font)
        y += 70
    png_path = HERE / "_scan_tmp.png"
    img.save(png_path)

    c = canvas.Canvas(str(out), pagesize=A4)
    c.drawImage(str(png_path), 0, 0, width=A4[0], height=A4[1])
    c.save()
    png_path.unlink(missing_ok=True)  # keep only the PDF
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    build_table_pdf(HERE / "sample_table.pdf")
    build_scanned_pdf(HERE / "sample_scanned.pdf")
