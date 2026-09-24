// Deck sesi "Exam Preparation — dan Langkah Setelahnya" (batch-agnostic).
// Build: NODE_PATH=$(npm root -g) node build_exam_prep.js
// Fakta NVIDIA dicek 24 September 2026; cek ulang halaman resmi sebelum dipakai lagi.
const pptxgen = require("pptxgenjs");
const path = require("path");

const C = { bg: "1A1A2E", card: "2D2D44", dark: "23233A", green: "76B900",
  lime: "A3D944", white: "FFFFFF", gray: "AAAACC", red: "EF5350",
  orange: "FF6F00", blue: "42A5F5" };
const FONT = "Calibri";
const CHECKED = "Dicek 24 September 2026.";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Exam Preparation — dan Langkah Setelahnya";
let slideNo = 0;

function base(title, section) {
  const s = pres.addSlide();
  slideNo += 1;
  s.background = { color: C.bg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 0.08, fill: { color: C.green } });
  if (section) s.addText(section, { x: 0.6, y: 0.35, w: 12, h: 0.3, margin: 0,
    fontFace: FONT, fontSize: 13, color: C.green, bold: true });
  if (title) s.addText(title, { x: 0.6, y: 0.7, w: 12.1, h: 0.75, margin: 0,
    fontFace: FONT, fontSize: 30, color: C.white, bold: true });
  s.addText(`Navasena · Exam Preparation   ${slideNo}`, { x: 8.7, y: 7.05, w: 4.1, h: 0.3,
    margin: 0, align: "right", fontFace: FONT, fontSize: 10, color: C.gray });
  return s;
}

// Array baris → paragraf terpisah (tiap baris satu <a:p>).
function lines(arr, extra = {}) {
  return arr.map((t, i) => ({ text: t, options: { breakLine: i < arr.length - 1, ...extra } }));
}

function bulletList(s, items, box, fontSize = 20, space = 12) {
  s.addText(lines(items, { bullet: true, paraSpaceAfter: space }),
    { ...box, fontFace: FONT, fontSize, color: C.white, valign: "top" });
}

function card(s, { x, y, w, h, head, body, accent = C.green, bodySize = 15 }) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: C.card }, line: { color: accent, width: 1.5 } });
  s.addText(head, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.45, margin: 0,
    fontFace: FONT, fontSize: 18, bold: true, color: accent });
  s.addText(lines(body, { paraSpaceAfter: 6 }), { x: x + 0.25, y: y + 0.75, w: w - 0.5,
    h: h - 0.9, margin: 0, fontFace: FONT, fontSize: bodySize, color: C.white, valign: "top" });
}

function callout(s, text, y = 6.2, color = C.lime) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 0.08, h: 0.55, fill: { color } });
  s.addText(text, { x: 0.85, y, w: 11.9, h: 0.55, margin: 0, valign: "middle",
    fontFace: FONT, fontSize: 17, color, bold: true });
}

function table(s, rows, colW, y = 1.75, fontSize = 15, rowH = undefined) {
  const head = rows[0].map(t => ({ text: t, options: { bold: true, color: C.bg, fill: { color: C.green } } }));
  const body = rows.slice(1).map((r, i) => r.map(cell => {
    const o = typeof cell === "object" ? cell : { text: cell };
    return { text: o.text, options: { color: o.color || C.white, bold: !!o.color,
      fill: { color: i % 2 ? C.dark : C.card } } };
  }));
  s.addTable([head, ...body], { x: 0.6, y, w: colW.reduce((a, b) => a + b, 0), colW,
    fontFace: FONT, fontSize, valign: "middle", margin: 0.08, rowH,
    border: { type: "solid", pt: 0.5, color: C.bg } });
}

// ── 1. Judul ────────────────────────────────────────────────
{
  const s = base(null, null);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.2, w: 0.12, h: 2.3, fill: { color: C.green } });
  s.addText("Exam Preparation", { x: 0.95, y: 2.1, w: 11.5, h: 1.1, margin: 0,
    fontFace: FONT, fontSize: 54, bold: true, color: C.white });
  s.addText("— dan langkah setelahnya", { x: 0.95, y: 3.15, w: 11.5, h: 0.6, margin: 0,
    fontFace: FONT, fontSize: 28, color: C.lime });
  s.addText("Lulus NCA-GENL, lalu ke mana?", { x: 0.95, y: 3.9, w: 11.5, h: 0.5, margin: 0,
    fontFace: FONT, fontSize: 20, color: C.gray });
  s.addText("Navasena · NCA-GENL Bootcamp", { x: 0.95, y: 6.3, w: 8, h: 0.4, margin: 0,
    fontFace: FONT, fontSize: 16, color: C.gray });
  s.addNotes(
    "Sesi ini dibagi tiga bagian. Pertama, cara lulus ujian NCA-GENL: formatnya, domain mana " +
    "yang sudah tercakup bootcamp, dan apa yang masih perlu kamu pelajari sendiri. Kedua, apa saja " +
    "yang disediakan NVIDIA di luar ujian: sertifikasi lanjutan, kursus DLI, Developer Program, " +
    "dan komunitasnya. Ketiga, arah karier dan cara mengubah capstone jadi portofolio.\n\n" +
    "Pesan utamanya: sertifikat itu satu langkah, bukan tujuan akhir. Kita ingin kamu pulang " +
    "dengan langkah berikutnya yang jelas, bukan hanya daftar tips ujian.");
}

// ── 2. Posisi kamu sekarang ─────────────────────────────────
{
  const s = base("Posisi kamu sekarang", "Pembuka");
  const done = [
    ["Model klasik", "Regresi, klasifikasi, clustering, ensemble, time series"],
    ["Deep learning", "Neural network, CNN, LSTM, transfer learning di GPU"],
    ["Transformer & LLM", "Transformer dari nol, fine-tune LoRA, evaluasi, SLM"],
    ["RAG", "Chunking, reranking, evaluasi RAGAS, deploy sebagai API"],
    ["Ekosistem NVIDIA", "RAPIDS, TensorRT, Triton, NIM, NeMo Guardrails, Jetson"],
    ["Trustworthy AI", "Fairness, guardrails, privasi data"],
  ];
  done.forEach(([h, b], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    card(s, { x: 0.6 + col * 4.1, y: 1.75 + row * 2.05, w: 3.9, h: 1.85, head: h, body: [b],
      accent: C.green, bodySize: 18 });
  });
  callout(s, "Buka lagi hasil pre-test kamu: domain mana yang dulu paling lemah?");
  s.addNotes(
    "Mulai dari apa yang sudah kamu bangun, bukan dari apa yang belum. Enam kotak ini " +
    "merangkum isi modul 01 sampai 06. Hampir semua ini muncul di blueprint NCA-GENL.\n\n" +
    "[Arahan: minta peserta membuka hasil pre-test.] Buka lagi hasil pre-test kamu. " +
    "Domain yang dulu nilainya paling rendah " +
    "dan sekarang sudah dibahas di bootcamp adalah bukti kemajuan. Domain yang dulu rendah dan " +
    "belum banyak dibahas adalah prioritas belajar berikutnya. Kita bahas petanya di slide 4.");
}

// ── 3. Format ujian ─────────────────────────────────────────
{
  const s = base("Format ujian dalam satu slide", "Bagian 1 · Lulus NCA-GENL");
  const facts = [
    ["50–60", "soal pilihan ganda"], ["60 menit", "durasi ujian"], ["$125", "biaya ujian"],
    ["Online", "proctored secara remote"], ["Inggris", "bahasa soal"], ["2 tahun", "masa berlaku"],
  ];
  facts.forEach(([big, small], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.75 + row * 2.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.9, h: 1.85, rectRadius: 0.08,
      fill: { color: C.card }, line: { color: C.card, width: 0 } });
    s.addText(big, { x, y: y + 0.3, w: 3.9, h: 0.8, margin: 0, align: "center",
      fontFace: FONT, fontSize: 36, bold: true, color: C.green });
    s.addText(small, { x, y: y + 1.1, w: 3.9, h: 0.45, margin: 0, align: "center",
      fontFace: FONT, fontSize: 17, color: C.gray });
  });
  callout(s, "Passing score tidak dipublikasikan NVIDIA. Abaikan angka dari sumber tak resmi.",
    6.2, C.orange);
  s.addNotes(
    "Semua angka di slide ini dari halaman resmi NCA-GENL. " + CHECKED + "\n" +
    "Sumber: https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/\n\n" +
    "Level Associate, entry-level. Pendaftaran lewat tombol Register for Exam di halaman itu " +
    "(platform Certiverse). Lulus mendapat digital badge dan sertifikat opsional. " +
    "Setelah dua tahun, sertifikasi diperbarui dengan mengikuti ujian lagi.\n\n" +
    "Soal passing score pasti ditanyakan. Jawabannya: NVIDIA tidak memublikasikannya. " +
    "Angka batas lulus yang beredar di internet tidak punya dasar resmi. Strategi yang " +
    "aman adalah menguasai materinya, bukan mengejar batas minimal. Satu lagi: " +
    "baca examination policy NVIDIA sebelum menjadwalkan ujian.");
}

// ── 4. Modul → domain ───────────────────────────────────────
{
  const s = base("Modul bootcamp → domain ujian", "Bagian 1 · Lulus NCA-GENL");
  const ok = { text: "Tercakup", color: C.green };
  const part = { text: "Sebagian", color: C.orange };
  table(s, [
    ["Domain ujian", "Bobot", "Dibahas di modul", "Status"],
    ["Core Machine Learning & AI Knowledge", "30%", "M01 ML, M02 DL, M04 transformer & LLM", ok],
    ["Software Development", "24%", "M04 produksi LLM, M05 RAG API, M06 serving", ok],
    ["Experimentation", "22%", "M04 LoRA & evaluasi, M05 RAGAS", part],
    ["Data Analysis & Visualization", "14%", "M01 EDA, M03 NLP, RAPIDS", ok],
    ["Trustworthy AI", "10%", "M06 fairness, guardrails, privasi", ok],
  ], [4.3, 1.1, 5.0, 1.7], 1.75, 19, 0.65);
  callout(s, "Celah terbesar ada di Experimentation: bobotnya 22%, tapi baru tercakup sebagian.");
  s.addNotes(
    "Bobot domain dari blueprint resmi NCA-GENL. " + CHECKED + "\n" +
    "Sumber: https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/\n\n" +
    "Status diambil dari isi notebook bootcamp. \"Tercakup\" artinya ada notebook yang membahas " +
    "topik itu secara langsung. Experimentation hanya sebagian: kita sudah mengevaluasi model " +
    "dan RAG, tapi belum membahas desain eksperimen, A/B test, atau experiment tracking. " +
    "Topik ini dan beberapa topik lain kita bahas di slide berikutnya.");
}

// ── 5. Topik tipis ──────────────────────────────────────────
{
  const s = base("Topik tipis: pelajari sendiri", "Bagian 1 · Lulus NCA-GENL");
  const gaps = [
    ["Alignment: RLHF & DPO", ["Kenapa model chat mengikuti instruksi.",
      "Reward model, data preferensi, beda RLHF dan DPO.", "Alignment disebut di topik resmi."], C.orange],
    ["Desain eksperimen", ["Baseline, satu variabel per percobaan.",
      "A/B test dan experiment tracking.", "Bagian dari domain Experimentation (22%)."], C.orange],
    ["NeMo Framework", ["Kamu sudah memakai NeMo Guardrails.",
      "Pelajari juga NeMo untuk training dan kustomisasi model."], C.blue],
    ["Distributed training", ["Kenapa model besar butuh banyak GPU.",
      "Data, tensor, dan pipeline parallelism.", "Cukup konsepnya, bukan implementasi."], C.blue],
  ];
  gaps.forEach(([h, b, a], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    card(s, { x: 0.6 + col * 6.15, y: 1.75 + row * 2.2, w: 5.95, h: 2.0, head: h, body: b,
      accent: a, bodySize: 16 });
  });
  s.addNotes(
    "Topik tipis adalah topik yang belum punya notebook sendiri di bootcamp. " +
    "Oranye untuk yang paling mungkin keluar di soal, biru untuk yang cukup dipahami konsepnya.\n\n" +
    "Alignment disebut langsung di daftar topik resmi NCA-GENL. Desain eksperimen bagian dari domain " +
    "Experimentation yang bobotnya 22%. NeMo Framework dan distributed training adalah konteks " +
    "ekosistem yang wajar dipahami: cukup tahu fungsinya dan kapan dipakai, bukan menulis kodenya.\n\n" +
    "Sumber belajar: cari di katalog DLI (ada filter Free Courses) dan cek silabusnya dulu sebelum " +
    "mendaftar. Sumber topik resmi: " +
    "https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/");
}

// ── 6. Urutan belajar ───────────────────────────────────────
{
  const s = base("Urutan belajar: mulai dari celah terbesar", "Bagian 1 · Lulus NCA-GENL");
  const steps = [
    ["Domain berbobot besar yang masih lemah", "Experimentation (22%), plus domain terlemahmu menurut pre-test."],
    ["Topik tipis", "Alignment, desain eksperimen, NeMo Framework, distributed training."],
    ["Latihan soal gaya ujian", "Soal berbahasa Inggris, pilih jawaban yang paling tepat."],
  ];
  steps.forEach(([h, b], i) => {
    const y = 1.85 + i * 1.5;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.1, w: 0.8, h: 0.8, fill: { color: C.green } });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.1, w: 0.8, h: 0.8, margin: 0, align: "center",
      valign: "middle", fontFace: FONT, fontSize: 26, bold: true, color: C.bg });
    s.addText(h, { x: 1.7, y: y + 0.05, w: 11, h: 0.45, margin: 0,
      fontFace: FONT, fontSize: 21, bold: true, color: C.white });
    s.addText(b, { x: 1.7, y: y + 0.5, w: 11, h: 0.4, margin: 0,
      fontFace: FONT, fontSize: 17, color: C.gray });
  });
  s.addNotes(
    "Urutannya disusun berdasarkan pengaruhnya ke skor, bukan dari yang paling mudah. Domain berbobot besar " +
    "yang masih lemah menyumbang paling banyak soal, jadi itu dulu.\n\n" +
    "Langkah pertama berbeda untuk tiap orang. Experimentation jadi prioritas kita semua karena " +
    "baru tercakup sebagian. Selain itu, hasil pre-test menunjukkan domain terlemahmu sendiri; " +
    "kalau itu Core ML yang bobotnya 30%, dahulukan juga.\n\n" +
    "Langkah terakhir penting karena soal ujian berbahasa Inggris. Membiasakan diri membaca soal " +
    "pilihan ganda teknis dalam bahasa Inggris sama pentingnya dengan menguasai materinya.\n\n" +
    "Bahan latihan yang sudah kamu punya: quiz tiap modul (file *-quiz.html di folder modul 01–06), " +
    "misalnya llm-fundamentals-quiz.html dan nvidia-ecosystem-quiz.html.");
}

// ── 7. Taktik hari ujian ────────────────────────────────────
{
  const s = base("Taktik di hari ujian", "Bagian 1 · Lulus NCA-GENL");
  bulletList(s, [
    "Baca examination policy NVIDIA dan siapkan perangkat sebelum jadwal ujian.",
    "Rata-rata sekitar satu menit per soal. Jangan terpaku di satu soal.",
    "Perhatikan kata kunci soal: best, most likely, first, NOT.",
    "Singkirkan pilihan yang jelas salah, lalu bandingkan sisanya.",
    "Pahami fungsi tiap produk NVIDIA: Triton, TensorRT, NeMo, NIM, RAPIDS.",
  ], { x: 0.6, y: 1.8, w: 12.1, h: 4.9 }, 24, 24);
  s.addNotes(
    "Taktik ini umum untuk ujian pilihan ganda berbatas waktu, disesuaikan dengan format NCA-GENL.\n\n" +
    "Satu menit per soal berasal dari 60 menit untuk 50 sampai 60 soal. Kalau satu soal terasa " +
    "buntu, pilih jawaban terbaik yang kamu punya lalu lanjut.\n\n" +
    "Kata kunci seperti \"most likely\" atau \"first\" sering membedakan dua pilihan yang " +
    "sama-sama benar secara teknis. Soal dengan \"NOT\" mudah terlewat kalau membaca terburu-buru.\n\n" +
    "Soal tentang produk NVIDIA kemungkinan besar menanyakan fungsinya: mana untuk serving, mana untuk optimasi " +
    "inferensi, mana untuk guardrails. Kamu sudah memakai hampir semuanya di bootcamp.\n\n" +
    "Latih tempo satu menit per soal dengan quiz modul (*-quiz.html) sebelum hari ujian.");
}

// ── 8. Jenjang sertifikasi ──────────────────────────────────
{
  const s = base("Jenjang sertifikasi NVIDIA", "Bagian 2 · Ekosistem NVIDIA");
  card(s, { x: 0.6, y: 1.75, w: 5.2, h: 4.2, head: "Associate", accent: C.green, bodySize: 16, body: [
    "NCA-GENL · Generative AI LLMs",
    "NCA-GENM · Multimodal Generative AI",
    "NCA-ADS · Accelerated Data Science",
    "NCA-AIIO · AI Infrastructure & Operations",
  ] });
  card(s, { x: 6.0, y: 1.75, w: 6.7, h: 4.2, head: "Professional", accent: C.blue, bodySize: 16, body: [
    "NCP-GENL · Generative AI LLMs",
    "NCP-AAI · Agentic AI",
    "NCP-ADS · Accelerated Data Science",
    "NCP-AIO · AI Operations",
    "NCP-AII · AI Infrastructure",
    "NCP-AIN · AI Networking",
    "NCP-ARI · AI Rack & Interconnect",
    "NCP-OUSD · OpenUSD Development",
  ] });
  callout(s, "Lanjutan paling dekat dari bootcamp ini: NCP-GENL dan NCP-AAI.");
  s.addNotes(
    "Daftar dari katalog sertifikasi resmi NVIDIA. " + CHECKED + "\n" +
    "Sumber: https://www.nvidia.com/en-us/learn/certification/\n\n" +
    "Associate menguji pemahaman konsep. Professional menguji kemampuan membangun dan " +
    "mengoperasikan sistem nyata, jadi butuh pengalaman praktik. Ujian Associate yang dicek " +
    "berbiaya $125. Harga Professional berbeda-beda per ujian, cek di halaman masing-masing.\n\n" +
    "NCP-GENL adalah versi lanjutan dari ujian yang sedang kita siapkan. NCP-AAI (Agentic AI) " +
    "cocok untuk yang ingin mendalami agen berbasis LLM, lanjutan alami dari modul RAG. " +
    "Katalog ini sering berubah, jadi selalu cek halaman resminya.");
}

// ── 9. DLI ──────────────────────────────────────────────────
{
  const s = base("NVIDIA DLI: kursus langsung dari sumbernya", "Bagian 2 · Ekosistem NVIDIA");
  card(s, { x: 0.6, y: 1.75, w: 5.95, h: 2.1, head: "Self-paced", accent: C.green, bodySize: 16, body: [
    "Online, bisa diambil kapan saja.",
    "Banyak kursus populer yang gratis.",
    "Kursus rekomendasi NCA-GENL: $90 per kursus, ±8 jam.",
  ] });
  card(s, { x: 6.75, y: 1.75, w: 5.95, h: 2.1, head: "Instructor-led workshop", accent: C.blue, bodySize: 16, body: [
    "Dipandu instruktur.",
    "Workshop rekomendasi: $500 per workshop, ±8 jam.",
  ] });
  card(s, { x: 0.6, y: 4.05, w: 12.1, h: 2.4, head: "Direkomendasikan di halaman NCA-GENL", accent: C.lime,
    bodySize: 16, body: [
      "Getting Started With Deep Learning · Accelerating End-to-End Data Science Workflows",
      "Introduction to Transformer-Based Natural Language Processing",
      "Building LLM Applications with Prompt Engineering",
      "Rapid Application Development With Large Language Models (LLMs)",
    ] });
  s.addNotes(
    "Deep Learning Institute adalah unit pelatihan resmi NVIDIA. " + CHECKED + "\n" +
    "Sumber: https://www.nvidia.com/en-us/training/self-paced-courses/ dan " +
    "https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/\n\n" +
    "Halaman self-paced punya filter Free Courses. Mulai dari yang gratis. " +
    "Harga $90 dan $500 di slide adalah harga kursus yang direkomendasikan untuk NCA-GENL, " +
    "bukan harga semua kursus DLI.\n\n" +
    "Di halaman NCA-GENL, NVIDIA memetakan kursus-kursus ini ke domain ujian. Sebelum membeli, " +
    "cek silabusnya: pilih yang menutup domain terlemahmu.");
}

// ── 10. Developer Program + build.nvidia.com ────────────────
{
  const s = base("Developer Program dan build.nvidia.com", "Bagian 2 · Ekosistem NVIDIA");
  card(s, { x: 0.6, y: 1.75, w: 5.95, h: 3.1, head: "NVIDIA Developer Program", accent: C.green, bodySize: 17, body: [
    "Gratis.",
    "Akses tools, SDK, dan bahan belajar.",
    "Forum dan komunitas developer.",
    "Kredit cloud dari Google Cloud dan NVIDIA setelah mencapai target belajar tertentu.",
  ] });
  card(s, { x: 6.75, y: 1.75, w: 5.95, h: 3.1, head: "build.nvidia.com", accent: C.blue, bodySize: 17, body: [
    "Katalog NIM API: coba model besar tanpa GPU sendiri.",
    "Endpoint-nya sama dengan yang kita pakai di modul LLM, RAG, dan NVIDIA.",
    "Ada juga blueprint: contoh aplikasi siap pakai.",
  ] });
  callout(s, "API key-nya bisa kamu pakai untuk prototipe proyek pribadi. Cek batas pemakaian di akunmu.", 5.2);
  s.addNotes(
    "Dua akses yang paling berguna setelah bootcamp selesai. " + CHECKED + "\n" +
    "Sumber: https://developer.nvidia.com/developer-program dan https://build.nvidia.com/\n\n" +
    "Developer Program gratis dan jadi pintu masuk ke banyak sumber daya NVIDIA lain. " +
    "Kredit cloud diberikan setelah pencapaian belajar tertentu, jadi syaratnya bisa berubah.\n\n" +
    "build.nvidia.com sudah kamu kenal: generator di modul RAG dan NVIDIA memanggil NIM lewat " +
    "endpoint integrate.api.nvidia.com. Artinya kamu tidak perlu GPU besar untuk mencoba " +
    "model 70B. Batas pemakaiannya bisa berubah, jadi cek di akun masing-masing.");
}

// ── 11. Komunitas & event ───────────────────────────────────
{
  const s = base("Komunitas dan event", "Bagian 2 · Ekosistem NVIDIA");
  const items = [
    ["GTC", ["Konferensi tahunan NVIDIA di San Jose.", "Replay sesinya bisa ditonton on-demand.",
      "Peserta on-site bisa ujian sertifikasi tanpa biaya."], C.green],
    ["Jetson AI Lab", ["Tutorial generative AI di perangkat edge.", "Lanjutan dari lab Jetson di modul NVIDIA.",
      "Ada server Discord untuk tanya jawab."], C.lime],
    ["Developer Forums", ["Tanya jawab teknis soal produk NVIDIA.", "Cari dulu: masalahmu mungkin sudah pernah dibahas."], C.blue],
    ["Inception", ["Program untuk startup AI.", "Boleh join walau belum memakai GPU NVIDIA.",
      "Kode kursus DLI gratis dan kredit cloud."], C.orange],
  ];
  items.forEach(([h, b, a], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    card(s, { x: 0.6 + col * 6.15, y: 1.75 + row * 2.3, w: 5.95, h: 2.1, head: h, body: b,
      accent: a, bodySize: 16 });
  });
  s.addNotes(
    CHECKED + " Sumber:\n" +
    "GTC: https://www.nvidia.com/gtc/\n" +
    "Jetson AI Lab: https://www.jetson-ai-lab.com/\n" +
    "Developer Forums: https://forums.developer.nvidia.com/\n" +
    "Inception: https://www.nvidia.com/en-us/startups/\n\n" +
    "Informasi ujian gratis untuk peserta GTC on-site ada di halaman training NVIDIA " +
    "(https://www.nvidia.com/en-us/training/). Ketentuannya bisa berubah tiap tahun.\n\n" +
    "Jetson AI Lab cocok untuk yang tertarik edge AI. Di modul NVIDIA kita sudah menjalankan " +
    "LLM di Jetson Orin Nano. Inception relevan kalau kamu sedang atau ingin membangun " +
    "startup; manfaat investor bergantung pada kelayakan.");
}

// ── 12. Arah peran ──────────────────────────────────────────
{
  const s = base("Pilih dulu arah peranmu", "Bagian 3 · Karier dan portofolio");
  table(s, [
    ["Peran", "Sertifikasi berikutnya", "Proyek pertama"],
    ["LLM / AI application engineer", "NCP-GENL, NCP-AAI", "Agen RAG dengan tool calling dan guardrails"],
    ["ML engineer (fine-tune & evaluasi)", "NCP-GENL", "Fine-tune SLM untuk satu tugas + laporan evaluasi"],
    ["Data scientist", "NCA-ADS → NCP-ADS", "Pipeline cuDF/cuML di data nyata + benchmark CPU vs GPU"],
    ["MLOps / AI infrastructure", "NCA-AIIO → NCP-AIO", "Serving model dengan Triton + pemantauan latensi"],
    ["Edge AI engineer", "Belajar lewat Jetson AI Lab", "Asisten lokal di Jetson dengan Ollama atau TensorRT"],
  ], [3.7, 3.2, 5.2], 1.75, 17, 0.7);
  callout(s, "Pilih satu peran dan satu proyek. Kedalaman lebih bernilai daripada daftar sertifikat.");
  s.addNotes(
    "Tabel ini bukan daftar wajib. Tujuannya membantu kamu memilih satu arah dulu.\n\n" +
    "Sertifikasi di kolom tengah mengacu ke katalog NVIDIA: " +
    "https://www.nvidia.com/en-us/learn/certification/ . Tanda panah artinya mulai dari Associate, " +
    "lalu naik ke Professional.\n\n" +
    "Proyek pertama sengaja dipilih yang bisa dimulai dari notebook bootcamp: agen RAG dari modul " +
    "RAG, fine-tune SLM dari modul LLM, RAPIDS dari modul 01 dan 03, Triton dari modul NVIDIA, " +
    "dan Jetson dari lab edge. Rekruter lebih mudah menilai satu proyek yang dalam dan terukur " +
    "daripada lima sertifikat tanpa bukti kerja.");
}

// ── 13. Capstone → portofolio ───────────────────────────────
{
  const s = base("Dari capstone ke portofolio publik", "Bagian 3 · Karier dan portofolio");
  const steps = [
    ["Repo yang rapi", ["README: masalah, cara menjalankan, dan hasilnya."]],
    ["Angka yang terukur", ["Metrik, data, dan perangkat yang dipakai, seperti di bootcamp."]],
    ["Demo siap coba", ["Hugging Face Space, notebook Colab, atau video pendek."]],
    ["Tulisan singkat", ["Apa yang berhasil, apa yang gagal, dan apa yang kamu pelajari."]],
  ];
  steps.forEach(([h, b], i) => {
    card(s, { x: 0.6 + i * 3.1, y: 1.9, w: 2.7, h: 2.6, head: h, body: b, accent: C.green, bodySize: 17 });
    if (i < 3) s.addText("→", { x: 0.6 + i * 3.1 + 2.7, y: 2.95, w: 0.4, h: 0.5, margin: 0,
      align: "center", fontFace: FONT, fontSize: 20, color: C.lime, bold: true });
  });
  callout(s, "Capstone-mu sudah jadi bahan portofolio. Tinggal dikemas dan dipublikasikan.", 5.0);
  s.addNotes(
    "Capstone sudah berisi kerja nyata; yang kurang biasanya hanya kemasannya.\n\n" +
    "Repo yang rapi dimulai dari README yang menjawab tiga pertanyaan: masalah apa, bagaimana " +
    "menjalankannya, dan apa hasilnya. Angka yang terukur adalah kebiasaan yang kita pakai " +
    "sepanjang bootcamp: setiap klaim performa disertai data dan perangkatnya.\n\n" +
    "Demo tidak harus rumit. Notebook Colab yang langsung bisa dijalankan sudah cukup. " +
    "Tulisan singkat, misalnya di LinkedIn atau blog, membantu orang lain memahami keputusan " +
    "teknis yang kamu ambil, termasuk yang gagal.");
}

// ── 14. Langkah berikutnya ──────────────────────────────────
{
  const s = base("Langkah berikutnya", "Penutup");
  bulletList(s, [
    "Bandingkan hasil pre-test dengan peta domain tadi.",
    "Pelajari topik tipis, lalu daftar ujian NCA-GENL.",
    "Daftar NVIDIA Developer Program (gratis).",
    "Pilih satu arah peran dan satu proyek pertama.",
    "Rapikan capstone jadi portofolio publik.",
  ], { x: 0.6, y: 1.8, w: 6.3, h: 4.8 }, 20);
  const links = [
    ["Sertifikasi NVIDIA", "nvidia.com/en-us/learn/certification", "https://www.nvidia.com/en-us/learn/certification/"],
    ["Kursus DLI", "nvidia.com/en-us/training/self-paced-courses", "https://www.nvidia.com/en-us/training/self-paced-courses/"],
    ["Developer Program", "developer.nvidia.com/developer-program", "https://developer.nvidia.com/developer-program"],
    ["NIM API", "build.nvidia.com", "https://build.nvidia.com/"],
  ];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.75, w: 5.5, h: 4.9, rectRadius: 0.08,
    fill: { color: C.card }, line: { color: C.lime, width: 1.5 } });
  links.forEach(([label, shown, url], i) => {
    const y = 2.0 + i * 1.0;
    s.addText(label, { x: 7.45, y, w: 5.0, h: 0.35, margin: 0, fontFace: FONT, fontSize: 15,
      bold: true, color: C.lime });
    s.addText([{ text: shown, options: { hyperlink: { url } } }], { x: 7.45, y: y + 0.35, w: 5.0,
      h: 0.4, margin: 0, fontFace: FONT, fontSize: 15, color: C.white });
  });
  s.addText("Detail tiap domain: deck sertifikasi NCA-GENL di folder materi bootcamp.", { x: 7.45, y: 5.85, w: 5.0,
    h: 0.6, margin: 0, fontFace: FONT, fontSize: 13, italic: true, color: C.gray });
  s.addNotes(
    "Lima langkah ini sengaja tanpa tanggal: urutannya yang penting, bukan kecepatannya.\n\n" +
    "Tautan di kanan bisa diklik saat deck dibagikan. Deck sertifikasi NCA-GENL (file " +
    "nca_genl_slides.pdf, di repo: docs/nca_genl_slides/) membahas " +
    "kelima domain ujian satu per satu dan cocok dipakai sebagai bahan belajar mandiri.\n\n" +
    "[Arahan: tutup sesi dengan pesan pembuka.] Ingat lagi pesan di awal: sertifikat itu satu langkah. Yang membuat " +
    "kamu dipercaya adalah proyek yang bisa ditunjukkan dan dijelaskan.");
}

pres.writeFile({ fileName: path.join(__dirname, "exam_prep.pptx") })
  .then(f => console.log("wrote", f));
