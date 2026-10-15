/* ============================================================================
 * CAVISA Admin · Panel Kontrol Dataset Server & Pelatihan Model
 * ========================================================================== */

const CFG = Object.assign({
  apiBase: "http://localhost:8000",
  StoreKeyTheme: "cavisa.tema.v2"
}, window.CAVISA_CONFIG || window.SIKSSP_CONFIG || {});

const API = (path) => `${CFG.apiBase.replace(/\/+$/, "")}${path}`;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

let toastTimer;
function toast(pesan) {
  const el = $("#toast");
  if (!el) return;
  el.textContent = pesan;
  el.hidden = false;
  requestAnimationFrame(() => el.classList.add("is-show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove("is-show");
    setTimeout(() => { el.hidden = true; }, 250);
  }, 2600);
}

async function bacaJson(res) {
  try { return await res.json(); }
  catch { return null; }
}

function ambilHitunganKelas(counts = {}) {
  let abnormal = 0, nonsperm = 0, normal = 0;
  for (const [k, v] of Object.entries(counts)) {
    const key = k.toLowerCase().replace(/[-_\s]/g, "");
    if (key.startsWith("normal")) normal += Number(v) || 0;
    else if (key.startsWith("abnormal")) abnormal += Number(v) || 0;
    else if (key.startsWith("non")) nonsperm += Number(v) || 0;
  }
  return { abnormal, nonsperm, normal };
}

/* ==========================================================================
 * 1. MODUL SERVER & MODEL
 * ========================================================================== */
const Server = {
  models: [],
  activeModel: "mlp",

  async cekStatus() {
    const chip = $("#serverStatusChip");
    const statConn = $("#statServerConn");
    const statModel = $("#statActiveModel");
    const statDevice = $("#statDevice");

    if (chip) chip.textContent = "Memeriksa…";

    try {
      const res = await fetch(API("/health"), { signal: AbortSignal.timeout(4000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await bacaJson(res);

      if (chip) {
        chip.textContent = "Server Online";
        chip.className = "chip chip--ok";
      }
      if (statConn) {
        statConn.textContent = "Online (Aktif)";
        statConn.className = "sstat__val sstat__val--ok";
      }
      if (statDevice) statDevice.textContent = (data.device || "cpu").toUpperCase();

      Server.activeModel = data.active_model || "mlp";
      if (statModel) statModel.textContent = Server.activeModel.toUpperCase();

      await Server.muatDaftarModel();
    } catch (err) {
      if (chip) {
        chip.textContent = "Server Offline";
        chip.className = "chip chip--req";
      }
      if (statConn) {
        statConn.textContent = "Offline (Terputus)";
        statConn.className = "sstat__val sstat__val--warn";
      }
      if (statModel) statModel.textContent = "—";
      const tbody = $("#modelsTableBody");
      if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--stop);">Gagal terhubung ke backend di ${CFG.apiBase}. Pastikan uvicorn berjalan.</td></tr>`;
    }
  },

  async muatDaftarModel() {
    try {
      const res = await fetch(API("/api/v1/models"));
      if (!res.ok) return;
      const data = await bacaJson(res);
      Server.models = data.models || [];
      Server.activeModel = data.active_model || Server.activeModel;

      const tbody = $("#modelsTableBody");
      if (!tbody) return;
      tbody.innerHTML = "";

      Server.models.forEach(m => {
        const tr = document.createElement("tr");
        const isActive = m.is_active;
        if (isActive) tr.className = "is-active";

        const statusBadge = m.is_loaded
          ? (isActive ? '<span class="breakdown-pill breakdown-pill--normal">Aktif (Default)</span>' : '<span class="breakdown-pill">Siap</span>')
          : '<span class="breakdown-pill breakdown-pill--nonsperm">Belum Termuat</span>';

        const btnAksi = isActive
          ? '<span style="font-size:12px; color:var(--ok); font-weight:700;">✓ Sedang Digunakan</span>'
          : `<button type="button" class="btn btn--ghost btn--sm btn-set-active" data-id="${m.id}">Jadikan Aktif</button>`;

        const fileBobot = `${m.id}_model.pkl`;

        tr.innerHTML = `
          <td><b>${m.name}</b> <code style="font-size:11px; color:var(--muted);">${m.id.toUpperCase()}</code></td>
          <td style="color:var(--text-2); font-size:12px;">${m.paper_table || "Skema Orange"}</td>
          <td><code style="font-size:11.5px;">${fileBobot}</code></td>
          <td>${statusBadge}</td>
          <td>${btnAksi}</td>
        `;
        tbody.appendChild(tr);
      });

      $$(".btn-set-active", tbody).forEach(btn => {
        btn.addEventListener("click", () => Server.setAktif(btn.dataset.id));
      });
    } catch (err) {
      console.warn("Gagal memuat daftar model:", err);
    }
  },

  async setAktif(modelId) {
    try {
      const res = await fetch(API(`/api/v1/models/active?model=${encodeURIComponent(modelId)}`), { method: "POST" });
      if (!res.ok) {
        const j = await bacaJson(res);
        throw new Error(j?.detail || "Gagal mengaktifkan model");
      }
      toast(`Model ${modelId.toUpperCase()} sekarang aktif sebagai model utama klasifikasi.`);
      await Server.cekStatus();
    } catch (err) {
      toast(`Gagal: ${err.message}`);
    }
  },

  async reloadDisk() {
    const btn = $("#btnReloadModels");
    if (btn) btn.disabled = true;
    try {
      const res = await fetch(API("/api/v1/models/reload"), { method: "POST" });
      if (!res.ok) throw new Error("Gagal reload model");
      const j = await bacaJson(res);
      toast(j?.message || "Seluruh file model berhasil dimuat ulang dari disk.");
      await Server.cekStatus();
    } catch (err) {
      toast(`Galat: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
    }
  }
};

/* ==========================================================================
 * 2. MODUL PENGATURAN PATH DATASET SERVER
 * ========================================================================== */
const Datasets = {
  list: [],

  async muat() {
    const chip = $("#totalDatasetsChip");
    if (chip) chip.textContent = "Memindai…";

    try {
      const res = await fetch(API("/api/v1/datasets"), { signal: AbortSignal.timeout(4000) });
      if (!res.ok) throw new Error("Gagal mengambil data dataset");
      const data = await bacaJson(res);
      Datasets.list = data.datasets || [];

      if (chip) chip.textContent = `${Datasets.list.length} Folder Dataset Server`;

      Datasets.renderKartu();
      Datasets.renderDropdown();
    } catch (err) {
      if (chip) chip.textContent = "Gagal memindai";
      toast("Gagal memindai folder dataset di server.");
    }
  },

  renderKartu() {
    const grid = $("#datasetCardsGrid");
    if (!grid) return;
    grid.innerHTML = "";

    Datasets.list.forEach(d => {
      const card = document.createElement("div");
      card.className = "dataset-card";

      const { abnormal, nonsperm, normal } = ambilHitunganKelas(d.counts || {});
      const existsBadge = d.exists
        ? `<span class="breakdown-pill breakdown-pill--normal">Tersedia (${d.total_images.toLocaleString("id-ID")} citra)</span>`
        : `<span class="breakdown-pill breakdown-pill--nonsperm">Tidak Ditemukan</span>`;

      card.innerHTML = `
        <div class="dataset-card__head">
          <div>
            <h4 class="dataset-card__title">${d.id}</h4>
            <span style="font-size:12px; color:var(--muted);">${d.name}</span>
          </div>
          ${existsBadge}
        </div>
        <div>
          <span style="font-size:11px; font-weight:600; color:var(--muted);">Path Server:</span>
          <div class="dataset-card__path">${d.path}</div>
        </div>
        <div class="dataset-card__breakdown">
          <span class="breakdown-pill breakdown-pill--abnormal">Abnormal: ${abnormal.toLocaleString("id-ID")}</span>
          <span class="breakdown-pill breakdown-pill--nonsperm">Non-Sperm: ${nonsperm.toLocaleString("id-ID")}</span>
          <span class="breakdown-pill breakdown-pill--normal">Normal: ${normal.toLocaleString("id-ID")}</span>
        </div>
        <div class="dataset-card__actions">
          <span style="font-size:11.5px; color:var(--muted);">3 subfolder standar</span>
          <button type="button" class="btn btn--ghost btn--sm btn-pilih-latih" data-id="${d.id}">
            Pilih untuk Pelatihan →
          </button>
        </div>
      `;
      grid.appendChild(card);
    });

    $$(".btn-pilih-latih", grid).forEach(b => {
      b.addEventListener("click", () => {
        const id = b.dataset.id;
        const sel = $("#selDataset");
        if (sel) {
          sel.value = id;
          Datasets.perbaruiStatistik();
          $("#sec-training")?.scrollIntoView({ behavior: "smooth", block: "start" });
          toast(`Dataset "${id}" dipilih untuk pelatihan.`);
        }
      });
    });
  },

  renderDropdown() {
    const sel = $("#selDataset");
    if (!sel) return;
    const nilaiLama = sel.value;

    sel.innerHTML = "";
    Datasets.list.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.id;
      const statusKet = d.exists ? `${d.total_images.toLocaleString("id-ID")} citra` : "tidak ditemukan";
      opt.textContent = `${d.id} · ${d.name} (${statusKet})`;
      sel.appendChild(opt);
    });

    const optCustom = document.createElement("option");
    optCustom.value = "__custom__";
    optCustom.textContent = "-- Masukkan Path Folder Kustom di Server --";
    sel.appendChild(optCustom);

    if (Datasets.list.some(d => d.id === nilaiLama)) {
      sel.value = nilaiLama;
    } else if (Datasets.list.some(d => d.id === "data_latih")) {
      sel.value = "data_latih";
    }

    Datasets.perbaruiStatistik();
  },

  perbaruiStatistik() {
    const sel = $("#selDataset");
    const wrapCustom = $("#customFolderWrap");
    if (!sel) return;

    const val = sel.value;
    if (val === "__custom__") {
      if (wrapCustom) wrapCustom.hidden = false;
      $("#dstatTotal").textContent = "—";
      $("#dstatAbnormal").textContent = "—";
      $("#dstatNonsperm").textContent = "—";
      $("#dstatNormal").textContent = "—";
      return;
    }

    if (wrapCustom) wrapCustom.hidden = true;
    const d = Datasets.list.find(item => item.id === val);
    if (d) {
      const { abnormal, nonsperm, normal } = ambilHitunganKelas(d.counts || {});
      $("#dstatTotal").textContent = (d.total_images || 0).toLocaleString("id-ID");
      $("#dstatAbnormal").textContent = abnormal.toLocaleString("id-ID");
      $("#dstatNonsperm").textContent = nonsperm.toLocaleString("id-ID");
      $("#dstatNormal").textContent = normal.toLocaleString("id-ID");
    } else {
      $("#dstatTotal").textContent = "0";
      $("#dstatAbnormal").textContent = "0";
      $("#dstatNonsperm").textContent = "0";
      $("#dstatNormal").textContent = "0";
    }
  }
};

/* ==========================================================================
 * 3. MODUL PELATIHAN MODEL (TRAINING ENGINE)
 * ========================================================================== */
const Training = {
  intervalPoll: null,
  isTraining: false,

  async mulai() {
    const sel = $("#selDataset");
    let folder = sel ? sel.value : "data_latih";
    if (folder === "__custom__") {
      const kustom = $("#inpCustomFolder")?.value.trim() || $("#inpCustomPath")?.value.trim();
      if (!kustom) {
        toast("Silakan ketik path folder dataset kustom.");
        $("#inpCustomFolder")?.focus();
        return;
      }
      folder = kustom;
    }

    const modeVal = $("#selTrainMode")?.value;
    const maxPerClass = (modeVal && modeVal !== "all") ? parseInt(modeVal, 10) : null;
    const testSize = parseFloat($("#selTestSplit")?.value || "0.2");

    const checkedModels = $$('input[name="trainModels"]:checked').map(el => el.value);
    if (!checkedModels.length) {
      toast("Pilih minimal satu algoritma learner untuk dilatih.");
      return;
    }

    const btnStart = $("#btnStartTrain");
    const btnRefresh = $("#btnRefreshDatasets");
    if (btnStart) btnStart.disabled = true;
    if (btnRefresh) btnRefresh.disabled = true;

    $("#trainLiveBox").hidden = false;
    $("#trainResultsBox").hidden = true;
    $("#resultsEmpty").hidden = true;
    $("#trainStatusChip").textContent = "Sedang Melatih…";
    $("#trainStatusChip").className = "chip chip--req";

    $("#trainProgressFill").style.width = "0%";
    $("#trainLivePercent").textContent = "0%";
    $("#trainLiveStage").textContent = "Memulai pelatihan di server…";
    $("#trainLiveMeta").textContent = `Folder: ${folder} · Menghubungi backend…`;

    try {
      const payload = {
        dataset_folder: folder,
        max_per_class: maxPerClass,
        models: checkedModels,
        test_size: testSize
      };

      const res = await fetch(API("/api/v1/train"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const j = await bacaJson(res);
        throw new Error(j?.detail || "Gagal memulai pelatihan.");
      }

      toast("Proses pelatihan model dimulai di background server.");
      Training.isTraining = true;
      Training.mulaiPolling();
    } catch (e) {
      toast(`Gagal: ${e.message}`);
      $("#trainLiveBox").hidden = true;
      $("#trainStatusChip").textContent = "Gagal";
      $("#trainStatusChip").className = "chip chip--req";
      if (btnStart) btnStart.disabled = false;
      if (btnRefresh) btnRefresh.disabled = false;
    }
  },

  mulaiPolling() {
    clearInterval(Training.intervalPoll);
    Training.intervalPoll = setInterval(async () => {
      try {
        const res = await fetch(API("/api/v1/train/status"));
        if (!res.ok) return;
        const status = await bacaJson(res);
        if (!status) return;

        Training.perbaruiUIStatus(status);

        if (!status.is_training) {
          clearInterval(Training.intervalPoll);
          Training.intervalPoll = null;
          Training.isTraining = false;

          const btnStart = $("#btnStartTrain");
          const btnRefresh = $("#btnRefreshDatasets");
          if (btnStart) btnStart.disabled = false;
          if (btnRefresh) btnRefresh.disabled = false;

          if (status.error) {
            toast(`Pelatihan terhenti: ${status.error}`);
            $("#trainLiveStage").textContent = `Gagal: ${status.error}`;
            $("#trainStatusChip").textContent = "Error";
            $("#trainStatusChip").className = "chip chip--req";
          } else {
            $("#trainLiveBox").hidden = true;
            $("#trainResultsBox").hidden = false;
            $("#resultsEmpty").hidden = true;
            $("#trainStatusChip").textContent = "Selesai";
            $("#trainStatusChip").className = "chip chip--ok";
            Training.renderHasil(status);
            toast("Pelatihan selesai! Model baru langsung aktif di backend.");
            Server.cekStatus();
          }
        }
      } catch (e) {
        console.warn("Polling status pelatihan terputus:", e);
      }
    }, 1200);
  },

  perbaruiUIStatus(status) {
    const fill = $("#trainProgressFill");
    const percentEl = $("#trainLivePercent");
    const stageEl = $("#trainLiveStage");
    const metaEl = $("#trainLiveMeta");

    const pct = Math.max(0, Math.min(100, status.progress_percent || 0));
    if (fill) fill.style.width = `${pct}%`;
    if (percentEl) percentEl.textContent = `${pct}%`;
    if (stageEl) stageEl.textContent = status.current_stage || "Sedang memproses…";

    const elapsed = status.elapsed_seconds ? status.elapsed_seconds.toFixed(1) : "0";
    const totalImg = status.total_images ? ` · ${status.total_images} citra` : "";
    if (metaEl) {
      metaEl.textContent = `Folder: ${status.dataset_used || "—"}${totalImg} · Waktu: ${elapsed} dtk`;
    }
  },

  renderHasil(status) {
    const summary = $("#trainResultSummary");
    const grid = $("#modelMetricsGrid");
    const chip = $("#resultsStatusChip");
    if (!grid) return;

    const count = (status.saved_models || []).length;
    const elapsed = status.elapsed_seconds ? status.elapsed_seconds.toFixed(1) : "—";
    if (chip) chip.textContent = `${count} Model Dilatih`;
    if (summary) {
      summary.textContent = `${count} model berhasil dilatih dalam ${elapsed} dtk menggunakan dataset "${status.dataset_used}" (${status.total_images} citra). Model langsung aktif di backend.`;
    }

    grid.innerHTML = "";
    const results = status.results || {};
    const modelKeys = Object.keys(results);

    if (!modelKeys.length) {
      grid.innerHTML = `<p class="empty__hint">Model tersimpan: ${(status.saved_models || []).join(", ")}</p>`;
      return;
    }

    modelKeys.forEach(key => {
      const m = results[key];
      const card = document.createElement("div");
      card.className = "metric-card";

      const accPct = (m.accuracy * 100).toFixed(1);
      const f1Pct = (m.f1_macro * 100).toFixed(1);
      const precPct = (m.precision_macro * 100).toFixed(1);
      const recPct = (m.recall_macro * 100).toFixed(1);

      let cmHtml = "";
      if (m.confusion_matrix && m.classes && m.classes.length) {
        const cls = m.classes;
        const thead = cls.map(c => `<th>Pred ${c}</th>`).join("");
        let tbody = "";
        cls.forEach((actualClass, rowIdx) => {
          const rowVals = m.confusion_matrix[actualClass] || [];
          const cols = rowVals.map((val, colIdx) => {
            const isDiag = rowIdx === colIdx ? ' class="cm-diag"' : "";
            return `<td${isDiag}>${val}</td>`;
          }).join("");
          tbody += `<tr><th>Aktual ${actualClass}</th>${cols}</tr>`;
        });
        cmHtml = `
          <div class="cm-wrap">
            <table class="cm-table">
              <thead><tr><th></th>${thead}</tr></thead>
              <tbody>${tbody}</tbody>
            </table>
          </div>
        `;
      }

      card.innerHTML = `
        <div class="metric-card__head">
          <span class="metric-card__name">${m.name} (${m.model_id.toUpperCase()})</span>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="metric-card__ref">${m.paper_table || "Skema Orange"}</span>
            <button type="button" class="btn btn--ghost btn--sm btn-apply-model" data-id="${m.model_id}">
              Gunakan Model Ini
            </button>
          </div>
        </div>
        <div class="metric-card__stats">
          <div class="mstat"><span class="mstat__val">${accPct}%</span><span class="mstat__lbl">Akurasi</span></div>
          <div class="mstat"><span class="mstat__val">${f1Pct}%</span><span class="mstat__lbl">F1-Score</span></div>
          <div class="mstat"><span class="mstat__val">${precPct}%</span><span class="mstat__lbl">Presisi</span></div>
          <div class="mstat"><span class="mstat__val">${recPct}%</span><span class="mstat__lbl">Recall</span></div>
        </div>
        ${cmHtml}
      `;
      grid.appendChild(card);
    });

    $$(".btn-apply-model", grid).forEach(b => {
      b.addEventListener("click", () => Server.setAktif(b.dataset.id));
    });
  },

  async cekStatusAwal() {
    try {
      const res = await fetch(API("/api/v1/train/status"));
      if (!res.ok) return;
      const status = await bacaJson(res);
      if (!status) return;

      if (status.is_training) {
        $("#trainLiveBox").hidden = false;
        $("#trainResultsBox").hidden = true;
        $("#resultsEmpty").hidden = true;
        const btnStart = $("#btnStartTrain");
        if (btnStart) btnStart.disabled = true;
        Training.perbaruiUIStatus(status);
        Training.mulaiPolling();
      } else if (status.results && Object.keys(status.results).length > 0) {
        $("#trainResultsBox").hidden = false;
        $("#resultsEmpty").hidden = true;
        Training.renderHasil(status);
      }
    } catch {}
  }
};

/* ==========================================================================
 * 4. INISIALISASI ADMIN PANEL
 * ========================================================================== */
function inisialisasiAdmin() {
  /* --- Tema --- */
  let temaSimpan = null;
  try { temaSimpan = localStorage.getItem(CFG.StoreKeyTheme); } catch {}
  const prefersGelap = window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = temaSimpan || (prefersGelap ? "dark" : "light");
  $("#btnTheme")?.addEventListener("click", () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    localStorage.setItem(CFG.StoreKeyTheme, t);
  });

  /* --- Event Listeners Server & Datasets --- */
  $("#btnCheckHealth")?.addEventListener("click", () => {
    Server.cekStatus();
    toast("Status server diperiksa.");
  });
  $("#btnReloadModels")?.addEventListener("click", Server.reloadDisk);
  $("#btnScanDatasets")?.addEventListener("click", () => {
    Datasets.muat();
    toast("Memindai folder dataset server…");
  });
  $("#btnRefreshDatasets")?.addEventListener("click", () => {
    Datasets.muat();
    toast("Dataset disegarkan.");
  });
  $("#selDataset")?.addEventListener("change", Datasets.perbaruiStatistik);
  $("#btnStartTrain")?.addEventListener("click", Training.mulai);

  /* --- Test Custom Path --- */
  $("#btnTestCustomPath")?.addEventListener("click", () => {
    const val = $("#inpCustomPath")?.value.trim();
    if (!val) {
      toast("Masukkan path folder dataset.");
      return;
    }
    const sel = $("#selDataset");
    if (sel) {
      sel.value = "__custom__";
      Datasets.perbaruiStatistik();
      const inp = $("#inpCustomFolder");
      if (inp) inp.value = val;
      $("#sec-training")?.scrollIntoView({ behavior: "smooth", block: "start" });
      toast(`Path kustom disetel: ${val}`);
    }
  });

  /* --- Smooth Scroll & IntersectionObserver --- */
  const tautan = $$(".tab, .nav-link");
  const kartu = $$(".card");
  const IO = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      const id = en.target.id;
      tautan.forEach(t => {
        const href = t.getAttribute("href");
        t.classList.toggle("is-active", href === "#" + id);
      });
    });
  }, { rootMargin: "-30% 0px -40% 0px" });
  kartu.forEach(c => IO.observe(c));

  tautan.forEach(t => {
    t.addEventListener("click", (e) => {
      const href = t.getAttribute("href");
      if (href && href.startsWith("#")) {
        const targetEl = $(href);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
          history.replaceState(null, null, href);
        }
      }
    });
  });

  /* --- Inisialisasi Data Awal --- */
  Server.cekStatus();
  Datasets.muat();
  Training.cekStatusAwal();
}

document.addEventListener("DOMContentLoaded", inisialisasiAdmin);
