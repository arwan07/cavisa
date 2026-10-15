/* ============================================================================
 * CAVISA — Computer-Aided Visual Sperm Analysis
 * Front-end mobile-first (vanilla JS, tanpa dependensi)
 *
 * Struktur proyek:
 *   Nalatico/
 *   ├── Backend/     → FastAPI + SqueezeNet Orange + 5 ML Learners
 *   └── Frontend/    → berkas ini (index.html, assets/)
 *
 * Struktur berkas:
 *   assets/js/config.js → konfigurasi koneksi backend (dimuat lebih dulu)
 *   assets/js/app.js    → seluruh logika aplikasi
 *
 * Bagian:
 *   1. Konfigurasi   → diturunkan dari config.js
 *   2. Utilitas
 *   3. State + Penyimpanan
 *   4. Citra: muat, pratinjau, thumbnail
 *   5. Mesin klasifikasi (backend API / analisis lokal)
 *   6. UI: hasil, riwayat, aksi
 *   7. Inisialisasi
 * ========================================================================== */

/* ==========================================================================
 * 1. KONFIGURASI
 * --------------------------------------------------------------------------
 * Diambil dari assets/js/config.js. Bila config.js belum dimuat, dipakai
 * nilai bawaan agar halaman tetap berfungsi.
 * ========================================================================== */
const CFG = Object.assign({
  apiBase: null,
  batasWaktuMs: 30000,
  maksUkuranMB: 10,
  maksBerkas: 12,
  maksBatch: 20,
  StoreKey: "cavisa.riwayat.v2",
  StoreKeyTheme: "cavisa.tema.v2",
  StoreKeyForm: "cavisa.form.v2",
  StoreKeyServer: "cavisa.server.v2"
}, window.CAVISA_CONFIG || window.SIKSSP_CONFIG || {});

const MODE_API = typeof CFG.apiBase === "string" && CFG.apiBase.trim().length > 0;
const API = (jalur) => `${CFG.apiBase.replace(/\/+$/, "")}${jalur}`;

/** Peta label backend ("Normal"/"Non-sperm") → kunci internal. */
const PETA_KELAS = {
  "normal": "normal",   "normal ": "normal",
  "abnormal": "abnormal",
  "nonsperm": "nonsperm", "non-sperm": "nonsperm", "non sperm": "nonsperm",
  "nonspermatozoa": "nonsperm"
};

function keKelasInternal(label) {
  return PETA_KELAS[String(label || "").trim().toLowerCase()] || null;
}

/* ==========================================================================
 * 2. UTILITAS
 * ========================================================================== */
const $  = (sel, akar = document) => akar.querySelector(sel);
const $$ = (sel, akar = document) => Array.from(akar.querySelectorAll(sel));

const LABELS = {
  normal:   { teks: "Normal",   warna: "ok" },
  abnormal: { teks: "Abnormal", warna: "warn" },
  nonsperm: { teks: "Non-Sperm",warna: "stop" }
};

/** Format tanggal Indonesia: "06 Oktober 2026" */
function formatTanggal(iso) {
  if (!iso) return "-";
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  const bulan = ["Januari","Februari","Maret","April","Mei","Juni",
                 "Juli","Agustus","September","Oktober","November","Desember"];
  return `${String(d).padStart(2,"0")} ${bulan[m-1]} ${y}`;
}

function formatWaktu(iso) {
  if (!iso) return "-";
  const dt = new Date(iso);
  if (isNaN(dt)) return "-";
  const jam = String(dt.getHours()).padStart(2,"0");
  const mnt = String(dt.getMinutes()).padStart(2,"0");
  return `${formatTanggal(iso)} · ${jam}:${mnt} WIB`;
}

function uid() {
  return "R" + Date.now().toString(36).toUpperCase().slice(-5) +
         Math.random().toString(36).slice(2, 5).toUpperCase();
}

function ukuranByte(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(0) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function jpegDataUrl(img, sisi = 160, kualitas = 0.72) {
  const c = document.createElement("canvas");
  c.width = c.height = sisi;
  const ctx = c.getContext("2d");
  const r = Math.min(img.width, img.height);
  ctx.drawImage(img, (img.width - r) / 2, (img.height - r) / 2, r, r, 0, 0, sisi, sisi);
  return c.toDataURL("image/jpeg", kualitas);
}

let toastTimer;
function toast(pesan) {
  const el = $("#toast");
  el.textContent = pesan;
  el.hidden = false;
  requestAnimationFrame(() => el.classList.add("is-show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove("is-show");
    setTimeout(() => { el.hidden = true; }, 250);
  }, 2600);
}

/* ==========================================================================
 * 3. STATE + PENYIMPANAN
 * ========================================================================== */
const State = {
  berkas: [],      // File + objectURL + thumbnail
  riwayat: [],     // rekaman hasil
  filter: "all",
  proses: false,
  sedangAnalisis: null
};

function muatStore(kunci, fallback) {
  try {
    const mentah = localStorage.getItem(kunci);
    return mentah ? JSON.parse(mentah) : fallback;
  } catch { return fallback; }
}
function simpanStore(kunci, nilai) {
  try { localStorage.setItem(kunci, JSON.stringify(nilai)); }
  catch { /* kuota penuh / mode privat — abaikan */ }
}

const Simpan = {
  riwayat: () => simpanStore(CFG.StoreKey, State.riwayat),
  bersihkan: () => { try { localStorage.removeItem(CFG.StoreKey); } catch {} },
  form: () => simpanStore(CFG.StoreKeyForm, {})
};

/* ==========================================================================
 * 4. CITRA: VALIDASI, MUAT, PRATINJAU
 * ========================================================================== */
const TIPE_DIITZINKAN = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/x-ms-bmp"];
const EKSTENSI_DIITZINKAN = /\.(jpe?g|png|bmp)$/i;

function validasiBerkas(file) {
  const maks = CFG.maksUkuranMB * 1024 * 1024;
  const tipeOk = TIPE_DIITZINKAN.includes(file.type);
  const extOk  = EKSTENSI_DIITZINKAN.test(file.name);
  if (!tipeOk && !extOk) return "Format tidak didukung. Gunakan JPG, PNG, atau BMP.";
  if (file.size > maks) return `Ukuran ${ukuranByte(file.size)} melebihi batas ${CFG.maksUkuranMB} MB.`;
  return null;
}

function muatGambar(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      try {
        if ("decode" in img) await img.decode();
      } catch {
        /* abaikan jika browser tidak mendukung decode() */
      }
      resolve({ img, url });
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Berkas citra rusak atau tidak dapat dibaca.")); };
    img.src = url;
  });
}

async function tambahBerkas(daftarFile) {
  const galat = $("#errFile");
  galat.hidden = true;
  let ditolak = 0;
  const berkasBaru = [];

  for (const file of daftarFile) {
    if (State.berkas.length >= CFG.maksBerkas) {
      galat.textContent = `Maksimal ${CFG.maksBerkas} citra per sesi.`;
      galat.hidden = false;
      break;
    }
    const pesan = validasiBerkas(file);
    if (pesan) {
      ditolak++;
      toast(`${file.name}: ${pesan}`);
      continue;
    }
    try {
      const { img, url } = await muatGambar(file);
      const item = {
        id: uid(),
        nama: file.name,
        ukuran: file.size,
        file,
        img,
        url,
        thumb: jpegDataUrl(img)
      };
      State.berkas.push(item);
      berkasBaru.push(item);
    } catch (e) {
      ditolak++;
      toast(e.message);
    }
  }

  if (ditolak) galat.textContent = `${ditolak} berkas ditolak.`;
  renderBerkas();
  if (berkasBaru.length) {
    // Mode server: jika banyak berkas diunggah sekaligus (batch)
    if (MODE_API && berkasBaru.length > 1) {
      toast(`${berkasBaru.length} citra dikirim ke server…`);
      await analisisSemuaBerkas(berkasBaru);
    } else {
      // Satu citra diunggah: langsung klasifikasi citra baru tersebut
      const target = berkasBaru[berkasBaru.length - 1];
      toast(`Menganalisis ${target.nama}…`);
      await analisisBerkas(target);
    }
  }
}

/** Analisis massal (hanya mode server). Semua berkas selesai, lalu tampil. */
async function analisisSemuaBerkas(daftar) {
  if (State.proses || !daftar.length) return;

  const nama = $("#inpName").value.trim();
  const tanggal = $("#inpDate").value;
  const kode = $("#inpSample").value.trim();
  if (!nama || nama.length < 2 || !tanggal) {
    $("#inpName").blur(); $("#inpDate").blur();
    toast("Lengkapi nama pengguna dan tanggal terlebih dahulu.");
    return;
  }
  Simpan.form({ nama, tanggal, kode });

  State.proses = true;
  $("#resultCard").hidden = true;
  $("#resultEmpty").hidden = true;
  $("#resultLoading").hidden = false;
  setProgress(20, `Mengirim ${daftar.length} citra ke server…`);

  const mulai = performance.now();
  let hasil;
  try {
    hasil = await analisisBatch(daftar, nama, tanggal, kode);
    setProgress(80, "Menerima hasil server…");
    await jeda(200);
  } catch (err) {
    State.proses = false;
    $("#resultLoading").hidden = true;
    $("#resultEmpty").hidden = false;
    toast(err.message || "Analisis batch gagal.");
    return;
  }
  setProgress(100, "Selesai");
  await jeda(250);

  const waktu = new Date().toISOString();
  const dipakai = new Set();
  const rekaman = hasil.map((h, i) => {
    // server membalas dengan nama berkas; cocokkan thumbnail ke berkas yang
    // benar bila urutannya berubah (mis. batch paralel)
    let b = daftar.find(x => x.nama === h.namaBerkas && !dipakai.has(x.id));
    if (!b) b = daftar[i] && !dipakai.has(daftar[i].id) ? daftar[i] : daftar.find(x => !dipakai.has(x.id));
    if (b) dipakai.add(b.id);
    return {
      id: uid(),
      nama, tanggal,
      kode: kode || "-",
      namaBerkas: h.namaBerkas || (b && b.nama) || "-",
      thumb: b ? b.thumb : "",
      kelas: h.kelas,
      keyakinan: h.keyakinan,
      probabilitas: h.probabilitas || {},
      catatan: h.catatan,
      kriteria: h.kriteria,
      ms: h.msServer != null ? Math.round(h.msServer) : Math.round(performance.now() - mulai),
      sumber: "server",
      waktu
    };
  });

  State.riwayat = [...rekaman, ...State.riwayat].slice(0, 200);
  Simpan.riwayat();
  State.proses = false;
  $("#resultLoading").hidden = true;
  renderHasil(rekaman[0]);
  renderRiwayat();
  toast(`${rekaman.length} citra selesai diklasifikasi.`);
}

function buangBerkas(id) {
  const i = State.berkas.findIndex(b => b.id === id);
  if (i === -1) return;
  if (State.sedangAnalisis === id) return;
  URL.revokeObjectURL(State.berkas[i].url);
  State.berkas.splice(i, 1);
  renderBerkas();
}

function renderBerkas() {
  const ul = $("#fileList");
  ul.innerHTML = "";
  for (const b of State.berkas) {
    const li = document.createElement("li");
    li.className = "fileitem";
    li.innerHTML = `
      <img src="${b.thumb}" alt="">
      <div class="fileitem__body">
        <p class="fileitem__name"></p>
        <p class="fileitem__meta">${ukuranByte(b.ukuran)}</p>
      </div>
      <button type="button" class="fileitem__x" aria-label="Hapus ${b.nama}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
      </button>`;
    $(".fileitem__name", li).textContent = b.nama;           // anti-XSS
    $(".fileitem__body", li).style.cursor = "pointer";
    $("img", li).style.cursor = "pointer";
    $(".fileitem__body", li).addEventListener("click", () => analisisBerkas(b));
    $("img", li).addEventListener("click", () => analisisBerkas(b));
    $(".fileitem__x", li).addEventListener("click", () => buangBerkas(b.id));
    ul.appendChild(li);
  }
}

/* ==========================================================================
 * 5. MESIN KLASIFIKASI
 * --------------------------------------------------------------------------
 * Pendekatan (mode lokal / demo):
 *   a. Ambil citra → skala 320px
 *   b. Ambang Otsu pada kanal nilai keabuan untuk memisahkan kepala sel
 *      (basal pekat, mis. pewarnaan Diff-Quik/H&E) dari latar terang
 *   c. Labeling komponen terhubung 8-arah
 *   d. Komponen disaring menjadi "kandidat sperma" berdasarkan ukuran,
 *      soliditas, dan elongasi kepala
 *   e. Kriteria WHO 2021 diperiksa pada kandidat terbaik → kelas keluaran
 *
 * Untuk penggunaan produksi, ganti `klasifikasi()` dengan panggilan API
 * (lihat bagian 1. Konfigurasi). Fungsi `analisisBackend()` sudah tersedia.
 * ========================================================================== */

/** Otsu threshold pada histogram 0..255. */
function otsu(hist, total) {
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, maksVar = -1, best = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > maksVar) { maksVar = v; best = t; }
  }
  return best;
}

/** Moment of order ke-2 untuk ovalitas & orientasi. */
function momen(_x, _y, w, h, piksel) {
  let m00 = 0, mu20 = 0, mu02 = 0, mu11 = 0, cx = 0, cy = 0;
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      if (!piksel[j * w + i]) continue;
      cx += i; cy += j; m00++;
    }
  }
  if (!m00) return null;
  cx /= m00; cy /= m00;
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      if (!piksel[j * w + i]) continue;
      const dx = i - cx, dy = j - cy;
      mu20 += dx * dx; mu02 += dy * dy; mu11 += dx * dy;
    }
  }
  mu20 /= m00; mu02 /= m00; mu11 /= m00;

  const tr = mu20 + mu02;
  const det = mu20 * mu02 - mu11 * mu11;
  const disc = Math.max(0, (tr * tr) / 4 - det);
  const l1 = tr / 2 + Math.sqrt(disc);   // sumbu utama
  const l2 = tr / 2 - Math.sqrt(disc);   // sumbu minor
  const rasio = l1 > 0 ? l2 / l1 : 0;

  // eccentricity → konversi ke "circularity" 1..0 (1 = bulat sempurna)
  const circularity = rasio <= 0 ? 0 : Math.max(0, Math.min(1, rasio * 1.45));
  return { area: m00, cx, cy, l1, l2, rasio, circularity };
}

/** Labeling komponen terhubung 8-arah → larik label. */
/** Erosi/dilasi morfologi biner dengan elemen structuring 3×3. */
function morfologi(mask, w, h, mode) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          const v = (nx < 0 || ny < 0 || nx >= w || ny >= h) ? 0 : mask[ny * w + nx];
          if (mode === "erosi") n += v; else n += (v ? 0 : 1);
        }
      }
      if (mode === "erosi") out[y * w + x] = n === 9 ? 1 : 0;
      else out[y * w + x] = n <= 4 ? 1 : 0;   // toleransi noise: toleran hingga 4 piksel
    }
  }
  return out;
}

function labelKomponen(mask, w, h) {
  const labels = new Int32Array(w * h).fill(0);
  const stack = new Int32Array(w * h);
  let n = 0;
  for (let p = 0; p < w * h; p++) {
    if (!mask[p] || labels[p]) continue;
    n++;
    let sp = 0;
    stack[sp++] = p;
    labels[p] = n;
    while (sp > 0) {
      const cur = stack[--sp];
      const y = (cur / w) | 0, x = cur - y * w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const np = ny * w + nx;
          if (mask[np] && !labels[np]) { labels[np] = n; stack[sp++] = np; }
        }
      }
    }
  }
  return { labels, jumlah: n };
}

/* Parameter acuan untuk citra sperma pada bidang pandang 320×320 px. */
const REF = { areaIdeal: 300, areaToleransi: 280, minArea: 14, maksArea: 2600 };

function analisisLokal(img) {
  const S = 320;
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const sisi = Math.min(img.width, img.height);
  ctx.drawImage(img, (img.width - sisi) / 2, (img.height - sisi) / 2, sisi, sisi, 0, 0, S, S);
  const data = ctx.getImageData(0, 0, S, S).data;

  // (b) kanal keabuan + histogram
  const gray = new Uint8Array(S * S);
  const hist = new Uint32Array(256);
  for (let p = 0; p < S * S; p++) {
    const r = data[p*4], g = data[p*4+1], b = data[p*4+2];
    const v = (0.299 * r + 0.587 * g + 0.114 * b) | 0;
    gray[p] = v; hist[v]++;
  }

  // (c) mask basal: piksel yang terpewarnai (lebih gelap dari latar)
  const thr = otsu(hist, S * S);
  const ambang = Math.max(30, Math.min(thr + 8, 230));
  const mask = new Uint8Array(S * S);
  let gelap = 0;
  for (let p = 0; p < S * S; p++) {
    if (gray[p] <= ambang) { mask[p] = 1; gelap++; }
  }
  const rasioGelap = gelap / (S * S);

  // (d) opening morfologi → membuang ekor tipis, menyisakan inti kepala
  const opened = morfologi(morfologi(mask, S, S, "erosi"), S, S, "dilasi");

  // (e) komponen terhubung pada hasil opening
  const { labels, jumlah } = labelKomponen(opened, S, S);
  const kandidat = [];
  for (let L = 1; L <= jumlah; L++) {
    const inti = new Uint8Array(S * S);
    let luasInti = 0;
    for (let p = 0; p < S * S; p++) if (labels[p] === L) { inti[p] = 1; luasInti++; }
    if (luasInti < REF.minArea || luasInti > REF.maksArea) continue;

    //momentumen pada inti (bentuk kepala)
    const mom = momen(0, 0, S, S, inti);
    if (!mom) continue;

    // luas penuh (kepala + ekor): flood fill dari inti pada mask asli
    const bx = Math.floor(mom.cx), by = Math.floor(mom.cy);
    const seed = (labels[by * S + bx] === L) ? (by * S + bx) : cariPikselInti(inti, S, S);
    if (seed < 0) continue;
    const penuh = floodFill(mask, S, S, seed);
    mom.rasioArea = mom.area;                    // luas kepala (inti)
    mom.areaPenuh = penuh.area;                  // kepala + ekor
    mom.panjangEkor = Math.max(0, penuh.area - mom.area);
    mom.rasioAreaRelatif = mom.area / REF.areaIdeal;
    mom.circularity = Math.max(0, Math.min(1, mom.rasio));
    kandidat.push(mom);
  }

  return { ambang, rasioGelap, totalKomponen: jumlah, kandidat, sisi: S };
}

function cariPikselInti(mask, w, h) {
  for (let p = 0; p < w * h; p++) if (mask[p]) return p;
  return -1;
}

/** Flood fill 4-arah pada mask biner. */
function floodFill(mask, w, h, seed) {
  const out = new Uint8Array(w * h);
  const stack = [seed];
  out[seed] = 1;
  let area = 0;
  while (stack.length) {
    const cur = stack.pop();
    area++;
    const y = (cur / w) | 0, x = cur - y * w;
    const nb = [[x-1,y],[x+1,y],[x,y-1],[x,y+1]];
    for (const [nx, ny] of nb) {
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const np = ny * w + nx;
      if (mask[np] && !out[np]) { out[np] = 1; stack.push(np); }
    }
  }
  return { mask: out, area };
}

/** Nilai fitur → criteria WHO sederhana (skala 0..1, 1 = paling ideal). */
function skorKriteria(k) {
  const luas  = 1 - Math.min(1, Math.abs(k.rasioArea - REF.areaIdeal) / REF.areaToleransi);
  const oval  = Math.max(0, Math.min(1, (k.circularity - 0.30) / 0.38));
  const rasio = Math.max(0, Math.min(1, 1 - Math.abs(k.rasio - 0.60) / 0.40));
  // ekor: panjang ekor minimal ~1,5× luas kepala agar morfologi utuh
  const ekor  = Math.max(0, Math.min(1, k.panjangEkor / (k.rasioArea * 1.5)));
  const kuantitas = Math.min(1, k.rasioAreaRelatif);
  return {
    luas:  Math.max(0, luas),
    oval:  Math.max(0, oval),
    rasio: Math.max(0, rasio),
    ekor:  Math.max(0, ekor),
    kuantitas: Math.max(0, kuantitas)
  };
}

function klasifikasiLokal(img) {
  const a = analisisLokal(img);
  const k = a.kandidat.sort((x, y) => y.rasioArea - x.rasioArea)[0];

  // --- Kasus Non-Sperm: tidak ada objek sel yang meyakinkan -------------
  if (!k || a.rasioGelap < 0.015 || a.rasioGelap > 0.80) {
    const kosong = a.rasioGelap < 0.015;
    return {
      kelas: "nonsperm",
      keyakinan: kosong ? 0.90 : 0.82,
      catatan: kosong
        ? "Tidak ditemukan objek sel pada citra. Area diperiksa kosong atau di luar bidang fokus — periksa pencahan dan pembuatan smear."
        : "Objek memenuhi bidang pandang (pewarnaan pekat / smear terlalu tebal). Ambil ulang preparat dengan lebih encer.",
      kriteria: [
        { ok: true,  teks: "Deteksi area basofil", ket: `${a.totalKomponen} komponen, ${(a.rasioGelap * 100).toFixed(1)}% bidang` },
        { ok: false, teks: "Keberadaan sperma",    ket: "Tidak ditemukan bentuk kepala + ekor yang sesuai" },
        { ok: false, teks: "Kualitas citra",       ket: kosong ? "Perlu pemeriksaan ulang" : "Gunakan enceran lebih encer" }
      ]
    };
  }

  const s = skorKriteria(k);

  // --- Kriteria WHO 2021 (kriteria utama + penunjang) ------------------
  const daftar = [
    { nama: "Bentuk kepala oval dan simetris",  ok: s.oval  >= 0.55, ket: `skor ${Math.round(s.oval * 100)}/100` },
    { nama: "Luas kepala normal",               ok: s.luas  >= 0.50, ket: `skor ${Math.round(s.luas * 100)}/100` },
    { nama: "Proporsi panjang-lebar kepala",    ok: s.rasio >= 0.50, ket: `skor ${Math.round(s.rasio * 100)}/100` },
    { nama: "Ekor teramati utuh",              ok: s.ekor  >= 0.40, ket: `skor ${Math.round(s.ekor * 100)}/100` },
    { nama: "Ukuran sel sesuai standar",        ok: s.kuantitas >= 0.70, ket: `${Math.round(k.rasioArea)} px (acuan ${REF.areaIdeal})` }
  ];
  const gagal = daftar.filter(g => !g.ok);
  const skorTotal = (s.oval + s.luas + s.rasio + s.ekor + s.kuantitas) / 5;

  if (gagal.length === 0) {
    return {
      kelas: "normal",
      keyakinan: Math.min(0.97, 0.70 + skorTotal * 0.27),
      catatan: "Sel sperma ditemukan dengan morfologi sesuai kriteria WHO 2021. Lanjutkan penentuan motilitas dan parameter lain.",
      kriteria: daftar.map(g => ({ ok: true, teks: g.nama, ket: g.ket }))
    };
  }

  return {
    kelas: "abnormal",
    keyakinan: Math.min(0.96, 0.62 + gagal.length * 0.10),
    catatan: `Terdeteksi ${gagal.length} dari ${daftar.length} kriteria morfologi tidak terpenuhi pada sel utama. Rujuk kepada pemeriksa laboratorium untuk verifikasi.`,
    kriteria: daftar.map(g => ({ ok: g.ok, teks: g.nama, ket: g.ket }))
  };
}

/**
 * Mengirim berkas citra asli (.bmp, .jpg, .jpeg, .png) langsung ke backend
 * tanpa konversi kanvas yang dapat menyebabkan race condition atau kompresi lossy.
 * Backend FastAPI + PIL kini mendukung .bmp secara native dengan presisi 100%.
 */
async function normalisasiUntukApi(berkas) {
  return { blob: berkas.file, nama: berkas.nama };
}

/** Error dengan status HTTP agar pesan ke pengguna lebih informatif. */
function httpError(status, pesan) {
  const e = new Error(pesan || `Server merespons HTTP ${status}.`);
  e.status = status;
  return e;
}

/** Baca body respons; backend FastAPI mengembalikan JSON pada 4xx/5xx. */
async function bacaJson(res) {
  try { return await res.json(); }
  catch { return null; }
}

async function pesanGalat(res, j) {
  const detail = j && j.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    // format validasi Pydantic v2
    return detail.map(d => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  }
  return null;
}

/* --------------------------------------------------------------------------
 * Pemeriksaan kesehatan server (GET /health)
 * -------------------------------------------------------------------------- */
async function cekKesehatan(jenis = "dalam") {
  if (!MODE_API) return { online: false, model: null };
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), jenis === "awal" ? 2500 : 6000);
    const res = await fetch(API("/health"), { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return { online: false, model: null };
    const j = await bacaJson(res) || {};
    return {
      online: true,
      model: j.model_loaded ?? j.model_ready ?? null,
      pesan: j.message || j.status || null
    };
  } catch {
    return { online: false, model: null };
  }
}

/* --------------------------------------------------------------------------
 * Prediksi tunggal (POST /api/v1/predict)
 * -------------------------------------------------------------------------- */
async function analisisBackend(berkas, nama, tanggal, kode) {
  const fd = new FormData();
  const kirim = await normalisasiUntukApi(berkas);
  fd.append("file", kirim.blob, kirim.nama);       // nama field sesuai Spec.md
  // konteks pemeriksaan dikirim sebagai form tambahan; backend boleh mengabaikan
  fd.append("nama_pengguna", nama);
  fd.append("tanggal_pemeriksaan", tanggal);
  if (kode) fd.append("kode_sampel", kode);

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), CFG.batasWaktuMs);
  let res;
  try {
    res = await fetch(API("/api/v1/predict"), { method: "POST", body: fd, signal: ctrl.signal });
  } catch (e) {
    if (e.name === "AbortError") throw httpError(0, "Server tidak merespons dalam batas waktu.");
    throw httpError(0, "Tidak dapat menghubungi server klasifikasi. Periksa apiBase di assets/js/config.js.");
  } finally { clearTimeout(t); }

  const j = await bacaJson(res);
  if (!res.ok) {
    throw httpError(res.status, (await pesanGalat(res, j)) || undefined);
  }

  const kelas = keKelasInternal(j.predicted_class);
  if (!kelas) throw httpError(500, `Label kelas tidak dikenal: "${j.predicted_class}".`);

  // Susun kriteria dari probabilitas per kelas supaya tetap informatif
  const prob = j.probabilities || {};
  const persen = (v) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "-");
  const kriteria = [
    { ok: true,  teks: "Prediksi model SqueezeNet + MLP", ket: `kelas "${j.predicted_class}"` },
    { ok: kelas !== "nonsperm", teks: "Sel sperma terdeteksi", ket: `Probabilitas ${persen(prob[Object.keys(prob).find(k => keKelasInternal(k) === kelas)])}` },
    { ok: true,  teks: "Waktu inferensi server",         ket: (j.processing_time_ms != null ? j.processing_time_ms + " ms" : "-") },
    { ok: true,  teks: "Berkas diproses",                 ket: j.filename || berkas.nama }
  ];

  return {
    kelas,
    keyakinan: typeof j.confidence === "number" ? Math.max(0, Math.min(1, j.confidence)) : 0.8,
    probabilitas: prob,
    msServer: j.processing_time_ms ?? null,
    catatan: (j.catatan || `Hasil dari server: ${j.predicted_class}.`) +
             " Rujuk ke pemeriksa laboratorium untuk konfirmasi.",
    kriteria
  };
}

/* --------------------------------------------------------------------------
 * Prediksi massal (POST /api/v1/predict-batch) — dipakai saat banyak berkas
 * diunggah sekaligus.
 * -------------------------------------------------------------------------- */
async function analisisBatch(daftarBerkas, nama, tanggal, kode) {
  const fd = new FormData();
  for (const b of daftarBerkas) {
    const k = await normalisasiUntukApi(b);
    fd.append("files", k.blob, k.nama);
  }
  fd.append("nama_pengguna", nama);
  fd.append("tanggal_pemeriksaan", tanggal);
  if (kode) fd.append("kode_sampel", kode);

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), CFG.batasWaktuMs * 2);
  let res;
  try {
    res = await fetch(API("/api/v1/predict-batch"), { method: "POST", body: fd, signal: ctrl.signal });
  } catch (e) {
    if (e.name === "AbortError") throw httpError(0, "Server tidak merespons dalam batas waktu.");
    throw httpError(0, "Tidak dapat menghubungi server klasifikasi. Periksa apiBase di assets/js/config.js.");
  } finally { clearTimeout(t); }

  const j = await bacaJson(res);
  if (!res.ok) throw httpError(res.status, (await pesanGalat(res, j)) || undefined);
  if (!Array.isArray(j)) throw httpError(500, "Format respons batch tidak dikenali.");

  return j.map((item, i) => {
    const kelas = keKelasInternal(item.predicted_class);
    return {
      kelas: kelas || "abnormal",
      keyakinan: typeof item.confidence === "number" ? Math.max(0, Math.min(1, item.confidence)) : 0.8,
      probabilitas: item.probabilities || {},
      msServer: item.processing_time_ms ?? null,
      namaBerkas: item.filename || (daftarBerkas[i] && daftarBerkas[i].nama),
      catatan: `Hasil dari server: ${item.predicted_class}.`,
      kriteria: [
        { ok: true, teks: "Prediksi model SqueezeNet + MLP", ket: `kelas "${item.predicted_class}"` },
        { ok: true, teks: "Waktu inferensi server", ket: (item.processing_time_ms != null ? item.processing_time_ms + " ms" : "-") }
      ]
    };
  });
}

/** Pemilih mode: backend API bila dikonfigurasi, selain itu mesin lokal. */
async function klasifikasi(berkas, nama, tanggal, kode) {
  if (MODE_API) return analisisBackend(berkas, nama, tanggal, kode);
  // jeda kecil supaya perubahan status terlihat di UI
  await new Promise(r => setTimeout(r, 900));
  return klasifikasiLokal(berkas.img);
}

/* ==========================================================================
 * 6. UI — PROGRESS, HASIL, RIWAYAT
 * ========================================================================== */
function setProgress(persen, teks) {
  $("#progressFill").style.width = persen + "%";
  if (teks) $("#progressText").textContent = teks;
}

function tampilLoading(aktif, persen, teks) {
  $("#resultLoading").hidden = !aktif;
  $("#resultEmpty").hidden = aktif || !$("#resultCard").hidden;
  if (aktif) { $("#resultCard").hidden = true; setProgress(persen, teks); }
}

async function analisisBerkas(berkas) {
  if (State.proses) return;

  // --- validasi formulir ---
  let valid = true;
  const nama = $("#inpName").value.trim();
  const tanggal = $("#inpDate").value;
  const kode = $("#inpSample").value.trim();

  const eN = $("#errName"), eD = $("#errDate");
  eN.hidden = eD.hidden = true;
  $("#inpName").removeAttribute("aria-invalid");
  $("#inpDate").removeAttribute("aria-invalid");

  if (nama.length < 2) {
    eN.textContent = "Nama pengguna minimal 2 karakter."; eN.hidden = false;
    $("#inpName").setAttribute("aria-invalid", "true"); valid = false;
  }
  if (!tanggal) {
    eD.textContent = "Tanggal pemeriksaan wajib diisi."; eD.hidden = false;
    $("#inpDate").setAttribute("aria-invalid", "true"); valid = false;
  }
  if (!valid) {
    $("#resultLoading").hidden = true;
    $("#resultCard").hidden = true;      // jangan tampilkan hasil lama
    $("#resultEmpty").hidden = false;
    toast("Lengkapi data pemeriksaan terlebih dahulu.");
    $("#h-form").scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }

  Simpan.form({ nama, tanggal, kode });

  // --- proses ---
  State.proses = true;
  State.sedangAnalisis = berkas.id;
  $("#resultCard").hidden = true;
  $("#resultEmpty").hidden = true;
  $("#resultLoading").hidden = false;
  setProgress(8, "Menyiapkan citra…");
  await jeda(120);
  setProgress(34, "Mensegmentasi citra…");
  await jeda(300);
  setProgress(62, "Mengekstraksi fitur morfologi…");
  await jeda(280);
  setProgress(88, "Menilai kriteria WHO…");
  await jeda(220);

  const mulai = performance.now();
  let keluar;
  try {
    keluar = await klasifikasi(berkas, nama, tanggal, kode);
  } catch (err) {
    State.proses = false;
    State.sedangAnalisis = null;
    tampilLoading(false);
    $("#resultEmpty").hidden = false;
    toast(err.message || "Analisis gagal.");
    return;
  }
  const durasi = Math.max(1, Math.round(performance.now() - mulai));

  setProgress(100, "Selesai");
  await jeda(280);

  const rekaman = {
    id: uid(),
    nama,
    tanggal,
    kode: kode || "-",
    namaBerkas: berkas.nama,
    thumb: berkas.thumb,
    kelas: keluar.kelas,
    keyakinan: keluar.keyakinan,
    probabilitas: keluar.probabilitas || {},
    catatan: keluar.catatan,
    kriteria: keluar.kriteria,
    ms: keluar.msServer != null ? Math.round(keluar.msServer) : durasi,
    sumber: MODE_API ? "server" : "lokal",
    waktu: new Date().toISOString()
  };

  State.riwayat.unshift(rekaman);
  State.riwayat = State.riwayat.slice(0, 200);
  Simpan.riwayat();

  State.proses = false;
  State.sedangAnalisis = null;
  $("#resultLoading").hidden = true;
  renderHasil(rekaman);
  renderRiwayat();
  toast("Klasifikasi selesai: " + LABELS[rekaman.kelas].teks);
}

const jeda = (ms) => new Promise(r => setTimeout(r, ms));

function renderHasil(r) {
  const l = LABELS[r.kelas];
  $("#resBadge").textContent = l.teks;
  $("#resBadge").className = "badge badge--" + r.kelas;
  $("#resConf").textContent = (r.keyakinan * 100).toFixed(1) + "%";
  $("#resImg").src = r.thumb;
  $("#resFile").textContent = r.namaBerkas;
  $("#resName").textContent = r.nama;
  $("#resDate").textContent = formatTanggal(r.tanggal);
  $("#resSample").textContent = r.kode;
  $("#resTime").textContent = r.ms + " ms";

  const ul = $("#resCriteria");
  ul.innerHTML = "";
  for (const k of (r.kriteria || [])) {
    const li = document.createElement("li");
    const tanda = k.ok ? "✓" : "✕";
    const kelasTanda = k.ok ? "dot--ok" : "dot--no";
    li.innerHTML = `<span class="dot ${kelasTanda}">${tanda}</span>
                    <span><b></b> — <span class="ket"></span></span>`;
    $("b", li).textContent = k.teks;
    $(".ket", li).textContent = k.ket || "";
    ul.appendChild(li);
  }

  $("#resNote").textContent = r.catatan || "";
  $("#resultCard").hidden = false;
  $("#resultEmpty").hidden = true;
}

function renderRiwayat() {
  const ul = $("#historyList");
  const data = State.filter === "all" ? State.riwayat : State.riwayat.filter(r => r.kelas === State.filter);

  $("#historyCount").textContent = State.riwayat.length;
  $("#historyEmpty").hidden = data.length > 0;
  ul.innerHTML = "";

  for (const r of data) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "histitem";
    btn.innerHTML = `
      <img src="${r.thumb}" alt="">
      <div class="histitem__body">
        <p class="histitem__name"></p>
        <p class="histitem__meta"></p>
      </div>
      <span class="histitem__tag tag--${r.kelas}">${LABELS[r.kelas].teks}</span>`;
    $(".histitem__name", btn).textContent = r.nama;
    $(".histitem__meta", btn).textContent = `${formatTanggal(r.tanggal)} · ${(r.keyakinan * 100).toFixed(0)}% · ${r.namaBerkas}`;
    btn.addEventListener("click", () => {
      renderHasil(r);
      $("#h-result").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

function eksporCSV() {
  if (!State.riwayat.length) { toast("Belum ada data untuk diekspor."); return; }
  const head = ["ID","Nama Pengguna","Tanggal","Kode Sampel","Berkas","Kelas","Keyakinan","Waktu Proses (ms)","Catatan"];
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const baris = State.riwayat.map(r =>
    [r.id, r.nama, r.tanggal, r.kode, r.namaBerkas, LABELS[r.kelas].teks,
     (r.keyakinan * 100).toFixed(1) + "%", r.ms, r.catatan].map(esc).join(",")
  );
  const csv = "﻿" + [head.map(esc).join(","), ...baris].join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `hasil-klasifikasi-sperma-${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast("Berkas CSV berhasil diunduh.");
}

/* ==========================================================================
 * 7. INISIALISASI
 * ========================================================================== */
function inisialisasi() {
  /* --- Tema --- */
  let temaSimpan = null;
  try { temaSimpan = localStorage.getItem(CFG.StoreKeyTheme); } catch {}
  const prefersGelap = window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = temaSimpan || (prefersGelap ? "dark" : "light");
  $("#btnTheme").addEventListener("click", () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    localStorage.setItem(CFG.StoreKeyTheme, t);
  });

  /* --- Tanggal default = hari ini --- */
  const hariIni = new Date();
  const offset = hariIni.getTimezoneOffset();
  $("#inpDate").value = new Date(hariIni.getTime() - offset * 60000).toISOString().slice(0, 10);

  /* --- Pulihkan formulir & riwayat --- */
  const f = (muatStore(CFG.StoreKeyForm, {}) || {});
  if (f.nama)  $("#inpName").value = f.nama;
  if (f.kode)  $("#inpSample").value = f.kode;
  if (f.tanggal) $("#inpDate").value = f.tanggal;
  State.riwayat = Array.isArray(muatStore(CFG.StoreKey, [])) ? muatStore(CFG.StoreKey, []) : [];
  renderRiwayat();

  /* --- Mode API + status server --- */
  const chip = $("#modeChip");
  const setChip = (teks, kelas) => { chip.textContent = teks; chip.className = kelas; };
  if (!MODE_API) {
    setChip("Mode demo · lokal", "chip chip--muted");
  } else {
    setChip("Memeriksa server…", "chip chip--muted");
    cekKesehatan("awal").then(k => {
      if (!k.online) setChip("Server offline", "chip chip--req");
      else if (k.model === false) setChip("Model belum siap", "chip chip--req");
      else setChip("Server aktif", "chip");
      simpanStore(CFG.StoreKeyServer, { online: k.online, model: k.model, cek: new Date().toISOString() });
    });
  }

  /* --- Upload: klik / drag-drop / paste --- */
  const dz = $("#dropzone"), inp = $("#inpFile");
  dz.addEventListener("click", () => inp.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inp.click(); }
  });
  inp.addEventListener("change", (e) => { tambahBerkas(Array.from(e.target.files)); e.target.value = ""; });

  ["dragenter", "dragover"].forEach(ev =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach(ev =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
  dz.addEventListener("drop", (e) => {
    const f = Array.from(e.dataTransfer?.files || []);
    if (f.length) tambahBerkas(f);
  });
  window.addEventListener("paste", (e) => {
    const f = Array.from(e.clipboardData?.files || []).filter(x => x.type.startsWith("image/"));
    if (f.length) tambahBerkas(f);
  });

  /* --- Validasi input langsung --- */
  $("#inpName").addEventListener("blur", (e) => {
    if (e.target.value.trim().length < 2 && e.target.value.length) {
      $("#errName").textContent = "Nama pengguna minimal 2 karakter.";
      $("#errName").hidden = false;
      e.target.setAttribute("aria-invalid", "true");
    } else {
      $("#errName").hidden = true;
      e.target.removeAttribute("aria-invalid");
    }
  });

  /* --- Filter riwayat --- */
  $$(".seg").forEach(b => b.addEventListener("click", () => {
    $$(".seg").forEach(x => x.classList.remove("is-active"));
    b.classList.add("is-active");
    State.filter = b.dataset.filter;
    renderRiwayat();
  }));

  /* --- Aksi --- */
  $("#btnReanalyze").addEventListener("click", () => {
    const terakhir = State.riwayat[0];
    const cocok = State.berkas.find(b => b.nama === terakhir?.namaBerkas);
    if (cocok) analisisBerkas(cocok);
    else if (State.berkas.length) analisisBerkas(State.berkas[State.berkas.length - 1]);
    else { toast("Unggah citra terlebih dahulu untuk dianalisis ulang."); $("#dropzone").click(); }
  });
  $("#btnPrint").addEventListener("click", () => window.print());
  $("#btnExport").addEventListener("click", eksporCSV);
  $("#btnClear").addEventListener("click", () => { $("#modal").hidden = false; });
  $$("#modal [data-close]").forEach(el => el.addEventListener("click", () => { $("#modal").hidden = true; }));
  $("#btnConfirmClear").addEventListener("click", () => {
    State.riwayat = [];
    Simpan.bersihkan();
    $("#modal").hidden = true;
    $("#resultCard").hidden = true;
    $("#resultEmpty").hidden = false;
    renderRiwayat();
    toast("Riwayat pemeriksaan dihapus.");
  });

  /* --- Tabbar & Desktop Nav: sorot sesuai posisi scroll & klik halus --- */
  const tautan = $$(".tab, .nav-link");
  const kartu = $$(".card");
  const IO = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      const targetId = en.target.getAttribute("aria-labelledby") || en.target.id;
      tautan.forEach(t => {
        const href = t.getAttribute("href");
        t.classList.toggle("is-active", href === "#" + targetId || href === "#" + en.target.id);
      });
    });
  }, { rootMargin: "-30% 0px -40% 0px" });
  kartu.forEach(c => IO.observe(c));

  tautan.forEach(t => {
    t.addEventListener("click", (e) => {
      const href = t.getAttribute("href");
      if (href && href.startsWith("#")) {
        const targetEl = $(href) || $(`[aria-labelledby="${href.slice(1)}"]`) || $(`#card${href.slice(1).replace(/^h-/, "").replace(/^[a-z]/, s => s.toUpperCase())}`);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
          history.replaceState(null, null, href);
        }
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", inisialisasi);