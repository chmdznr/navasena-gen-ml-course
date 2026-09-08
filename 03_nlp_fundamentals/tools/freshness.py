"""Cek apakah artefak hasil build (PDF) tertinggal dari sumbernya.

Dipisah ke modul sendiri karena dipakai validate_slides.py dan
validate_cheatsheet.py, dan alasannya cukup halus untuk tidak layak diduplikasi.
"""
import pathlib, subprocess


def tertinggal(artefak: pathlib.Path, sumber: pathlib.Path):
    """Kembalikan pesan error kalau `artefak` perlu dibangun ulang, atau None.

    Membandingkan mtime saja tidak cukup: `git checkout` menulis kedua berkas
    dalam detik yang sama dengan urutan sembarang, jadi artefak hasil clone
    bersih bisa terlihat "lebih tua" hanya karena selisih mikrodetik. Kalau
    sumbernya tidak berbeda dari yang sudah di-commit, artefaknya pasti ikut
    ter-commit bersamanya, jadi keduanya sinkron. Cek mtime hanya relevan saat
    sumbernya sedang diubah secara lokal.
    """
    if not artefak.exists():
        return f"{artefak.name} tidak ada"
    kotor = subprocess.run(["git", "status", "--porcelain", "--", str(sumber)],
                           capture_output=True, text=True, cwd=str(sumber.parent))
    if kotor.returncode == 0 and not kotor.stdout.strip():
        return None
    if artefak.stat().st_mtime < sumber.stat().st_mtime:
        return f"{artefak.name} lebih tua dari {sumber.name} yang sudah diubah — bangun ulang"
    return None


def demo():
    """Cek mandiri: berkas yang tidak ada selalu dilaporkan tertinggal."""
    assert tertinggal(pathlib.Path("/tidak/ada.pdf"), pathlib.Path(__file__)) is not None
    # Berkas yang sama dengan dirinya sendiri tidak pernah lebih tua.
    me = pathlib.Path(__file__)
    assert tertinggal(me, me) is None
    print("freshness.py OK")


if __name__ == "__main__":
    demo()
