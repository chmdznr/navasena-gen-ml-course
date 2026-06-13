# 05_rag/data/make_sample.py
"""Generate the baked-in sample corpus (sample_id_document.pdf).

Original Indonesian factual prose (paraphrased, not copied from any source) so the
notebooks always have a real multi-paragraph document to ingest even without an upload.
Run: python make_sample.py  ->  writes sample_id_document.pdf next to this file.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

PARAS = [
    "Candi Borobudur adalah candi Buddha terbesar di dunia yang terletak di "
    "Kabupaten Magelang, Jawa Tengah. Candi ini dibangun pada masa Wangsa Syailendra "
    "sekitar abad ke-8 hingga ke-9 Masehi.",
    "Bangunan Borobudur tersusun atas sembilan teras berundak: enam teras berbentuk "
    "persegi dan tiga teras berbentuk lingkaran. Di puncaknya terdapat sebuah stupa "
    "induk besar yang dikelilingi oleh stupa-stupa berlubang.",
    "Dinding candi dihiasi sekitar 2.672 panel relief dan terdapat 504 arca Buddha. "
    "Relief tersebut menceritakan ajaran Buddha serta kehidupan masyarakat pada masanya.",
    "Borobudur sempat terbengkalai dan tertutup material letusan gunung berapi. "
    "Candi ini ditemukan kembali pada tahun 1814 atas perhatian Thomas Stamford Raffles, "
    "lalu dipugar secara besar-besaran pada tahun 1975 hingga 1982 dengan dukungan UNESCO.",
    "Pada tahun 1991 Borobudur ditetapkan sebagai Situs Warisan Dunia UNESCO. "
    "Hingga kini candi ini menjadi salah satu tujuan wisata dan ziarah keagamaan "
    "yang paling banyak dikunjungi di Indonesia.",
]

def build(out: Path) -> None:
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Candi Borobudur")
    styles = getSampleStyleSheet()
    story = [Paragraph("Candi Borobudur", styles["Title"]), Spacer(1, 12)]
    for p in PARAS:
        story.append(Paragraph(p, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    print(f"wrote {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    build(Path(__file__).with_name("sample_id_document.pdf"))
