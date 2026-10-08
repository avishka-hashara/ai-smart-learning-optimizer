'use strict';

/* =========================================================================
   State
   ========================================================================= */
const state = {
  mode: null,          // 'user' (signed in) | 'demo' (personas, nothing saved)
  meta: null,          // /api/meta payload (demo)
  persona: null,       // active persona name (demo)
  user: null,          // /api/me
  userMeta: null,      // /api/me/meta: input ranges, current week
  subjects: [],        // the user's active subjects
  checkins: [],        // /api/checkins summaries, newest week first
  viewing: null,       // id of the check-in whose plan is on screen
  hours: [],           // inputs, aligned with inputSubjects()
  conf: [],
  logins: [],
  result: null,        // plan payload being shown
  running: false,
  activeTab: 'pred',
};

const DEFAULT_HOURS = 5;
const DEFAULT_CONF = 5;
const DIFFICULTY = { 1: 'Easy', 2: 'Medium', 3: 'Hard' };

// Subjects behind the input cards / the plan currently on screen
const inputSubjects = () => (state.mode === 'user' ? state.subjects : state.meta.subjects);
const resultSubjects = () => state.result.subject_meta;
const inputRanges = () => (state.mode === 'user' ? state.userMeta : state.meta);

const $ = (sel, root = document) => root.querySelector(sel);
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

/* =========================================================================
   Helpers
   ========================================================================= */
function el(tag, attrs = {}, html = '') {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'style' && typeof v === 'object') {
      for (const [prop, val] of Object.entries(v)) {
        if (prop.startsWith('--')) node.style.setProperty(prop, val);
        else node.style[prop] = val;
      }
    }
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  if (html) node.innerHTML = html;
  return node;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function toast(msg) {
  const node = el('div', { class: 'toast' }, esc(msg));
  $('#toasts').append(node);
  setTimeout(() => { node.classList.add('out'); setTimeout(() => node.remove(), 300); }, 2600);
}

async function api(path, body, method) {
  const opts = { method: method || (body === undefined ? 'GET' : 'POST') };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = res.status === 204 ? null : await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data?.detail)
      ? data.detail.map(d => `${d.loc?.slice(-1)[0] ?? ''}: ${d.msg}`.replace(/^: /, '')).join('; ')
      : data?.detail;
    const err = new Error(detail || `Request failed (${res.status})`);
    err.status = res.status;
    // Session expired or revoked mid-use: back to the sign-in screen
    if (res.status === 401 && state.mode === 'user' && !path.startsWith('/api/auth/')) {
      showAuth('login', 'Your session has ended. Please sign in again.');
    }
    throw err;
  }
  return data;
}

function fmtWeek(iso) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
}

function mondayOf(iso) {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

function daysUntil(iso) {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  return Math.round((new Date(`${iso}T00:00:00`) - today) / 86400000);
}

function scoreBand(v) {
  if (v >= 75) return { cls: 'good', label: 'Distinction', color: 'var(--good)' };
  if (v >= 50) return { cls: 'warn', label: 'Passing', color: 'var(--warn)' };
  return { cls: 'bad', label: 'At risk', color: 'var(--bad)' };
}

const LABEL_CLASS = { overconfident: 'bad', anxious: 'warn', aligned: 'good' };

/* =========================================================================
   Theme
   ========================================================================= */
function initTheme() {
  let theme = null;
  try { theme = localStorage.getItem('theme'); } catch { /* storage unavailable */ }
  if (!theme) theme = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  $('#themeBtn').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch { /* ignore */ }
    if (state.result) renderCharts();
  });
}

/* =========================================================================
   Sidebar
   ========================================================================= */
function renderPersonas() {
  const list = $('#personaList');
  list.innerHTML = '';
  for (const [name, p] of Object.entries(state.meta.personas)) {
    list.append(el('button', {
      class: 'persona',
      type: 'button',
      role: 'radio',
      'aria-checked': String(name === state.persona),
      onclick: () => selectPersona(name),
    }, `<span class="persona-avatar">${p.avatar}</span>
        <span><span class="persona-name">${esc(name.replace(/^The /, ''))}</span>
        <span class="persona-desc">${esc(p.description)}</span></span>`));
  }
}

function selectPersona(name) {
  state.persona = name;
  const p = state.meta.personas[name];
  state.hours = [...p.hours];
  state.conf = [...p.conf];
  renderPersonas();
  renderSubjects();
  closeSidebar();
  clearResults();
  if ($('#liveToggle').checked) runPipeline();
}

function openSidebar() { $('#sidebar').classList.add('open'); $('#scrim').hidden = false; }
function closeSidebar() { $('#sidebar').classList.remove('open'); $('#scrim').hidden = true; }

async function syncLms() {
  const btn = $('#syncBtn');
  btn.classList.add('is-loading');
  btn.disabled = true;
  try {
    state.logins = (await api('/api/lms-sync', {})).lms_logins;
    renderSubjects();
    toast('📡 LMS telemetry synced');
  } catch (e) {
    toast(`Sync failed: ${e.message}`);
  } finally {
    btn.classList.remove('is-loading');
    btn.disabled = false;
  }
}

/* =========================================================================
   Inputs
   ========================================================================= */
function setRangeFill(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty('--p', `${pct}%`);
}

const liveRun = debounce(() => runPipeline(), 600);

function onInputChanged() {
  renderBudget();
  if ($('#liveToggle').checked) liveRun();
}

function renderSubjects() {
  const grid = $('#subjectGrid');
  const [hMin, hMax] = inputRanges().hours_range;
  const [cMin, cMax] = inputRanges().conf_range;
  grid.innerHTML = '';

  inputSubjects().forEach((s, i) => {
    const pill = state.mode === 'demo'
      ? `<span class="pill" title="LMS logins this week">📡 ${state.logins[i]}</span>`
      : s.exam_date ? examPill(s.exam_date) : '';
    const card = el('div', { class: 'subject', style: { '--c': s.color } }, `
      <div class="subject-top">
        <span class="subject-name">${s.icon} ${esc(s.name)}</span>
        ${pill}
      </div>
      <div class="field">
        <label for="h${i}">Hours / week</label>
        <div class="field-row">
          <input type="range" id="h${i}" min="${hMin}" max="${hMax}" step="1" value="${state.hours[i]}">
          <input type="number" class="num" min="${hMin}" max="${hMax}" step="1" value="${state.hours[i]}" aria-label="${esc(s.name)} hours">
        </div>
      </div>
      <div class="field">
        <label for="c${i}">Confidence <output>${state.conf[i].toFixed(1)}</output></label>
        <input type="range" id="c${i}" min="${cMin}" max="${cMax}" step="0.5" value="${state.conf[i]}">
      </div>`);

    const hRange = $(`#h${i}`, card);
    const hNum = $('.num', card);
    const cRange = $(`#c${i}`, card);
    const cOut = $('output', card);

    hRange.addEventListener('input', () => {
      state.hours[i] = +hRange.value;
      hNum.value = hRange.value;
      setRangeFill(hRange);
      onInputChanged();
    });
    hNum.addEventListener('change', () => {
      const v = Math.min(hMax, Math.max(hMin, +hNum.value || hMin));
      state.hours[i] = v;
      hNum.value = v;
      hRange.value = v;
      setRangeFill(hRange);
      onInputChanged();
    });
    cRange.addEventListener('input', () => {
      state.conf[i] = +cRange.value;
      cOut.textContent = state.conf[i].toFixed(1);
      setRangeFill(cRange);
      onInputChanged();
    });

    setRangeFill(hRange);
    setRangeFill(cRange);
    grid.append(card);
  });
  renderBudget();
}

function renderBudget() {
  const total = state.hours.reduce((a, b) => a + b, 0);
  $('#budgetTotal').textContent = `${total.toFixed(0)} h`;
  const stack = $('#budgetStack');
  stack.innerHTML = '';
  inputSubjects().forEach((s, i) => {
    stack.append(el('span', {
      title: `${s.name}: ${state.hours[i]} h`,
      style: { flexGrow: state.hours[i], background: s.color },
    }));
  });
}

/* =========================================================================
   Pipeline run
   ========================================================================= */
function setStepper(stage) {
  // stage: 0 = idle, 1..4 = active stage, 5 = all done
  document.querySelectorAll('#stepper li').forEach(li => {
    const n = +li.dataset.stage;
    li.classList.toggle('done', n < stage);
    li.classList.toggle('active', n === stage);
  });
}

const runLabel = () => (state.mode === 'user' ? 'Save check-in' : 'Run pipeline');

async function runPipeline() {
  if (state.running || !state.mode) return;
  if (state.mode === 'user' && !state.subjects.length) return;
  state.running = true;
  const btn = $('#runBtn');
  btn.classList.add('is-loading');
  btn.disabled = true;
  $('.run-label', btn).textContent = 'Running…';
  $('#errorBox').hidden = true;

  // The API runs all stages in one call; walk the stepper while we wait.
  let stage = 1;
  setStepper(stage);
  const ticker = setInterval(() => { if (stage < 4) setStepper(++stage); }, 220);

  try {
    if (state.mode === 'user') {
      await saveCheckin();
    } else {
      state.result = await api('/api/run', { hours: state.hours, confidence: state.conf });
    }
    clearInterval(ticker);
    setStepper(5);
    renderResults();
    if (state.mode === 'user') toast(`✅ Check-in saved for the week of ${fmtWeek($('#weekInput').value)}`);
    else if (!$('#liveToggle').checked) toast(`✨ Pipeline finished in ${state.result.timings.total_ms} ms`);
  } catch (e) {
    clearInterval(ticker);
    setStepper(0);
    $('#errorBox').textContent = e.message;
    $('#errorBox').hidden = false;
  } finally {
    state.running = false;
    btn.classList.remove('is-loading');
    btn.disabled = false;
    $('.run-label', btn).textContent = runLabel();
  }
}

function clearResults() {
  state.result = null;
  $('#results').hidden = true;
  const noSubjects = state.mode === 'user' && !state.subjects.length;
  $('#emptyState').hidden = noSubjects;
  if (state.mode === 'user') {
    $('#emptyTitle').textContent = 'No plan for this week yet';
    $('#emptyText').textContent = 'Set how many hours you spent on each subject and how confident you feel, then save your check-in to get this week’s plan.';
  } else {
    $('#emptyTitle').textContent = 'Ready when you are';
    $('#emptyText').textContent = 'Run the pipeline to see predictions, risk diagnoses, the optimised hour split and your timetable.';
  }
  setStepper(0);
}

/* =========================================================================
   Results rendering
   ========================================================================= */
function renderResults() {
  $('#emptyState').hidden = true;
  $('#results').hidden = false;
  renderKpis();
  renderScores();
  renderRiskTable();
  renderExplanations();
  renderDeltas();
  renderBaselines();
  renderWeek();
  renderRaw();
  renderCharts();
}

function renderKpis() {
  const r = state.result;
  const subj = resultSubjects();
  const avg = r.predicted.reduce((a, b) => a + b, 0) / r.predicted.length;
  const top = r.risk.priority.indexOf(Math.max(...r.risk.priority));
  const bestBaseline = Math.max(...Object.values(r.allocation.baselines));
  const gain = r.allocation.fitness - bestBaseline;
  const band = scoreBand(avg);

  const kpis = [
    { label: 'Avg predicted score', value: `${avg.toFixed(1)}%`, color: band.color, desc: `${band.label} · ${subj.length} modules` },
    { label: 'Weekly budget', value: `${r.budget.toFixed(0)} h`, color: '#38bdf8', desc: `${(r.budget / 7).toFixed(1)} h per day` },
    { label: 'Top priority', value: `${subj[top].icon} ${subj[top].name}`, color: subj[top].color, desc: `Fuzzy priority ${r.risk.priority[top].toFixed(2)} / 10` },
    { label: 'GA fitness', value: r.allocation.fitness.toFixed(1), color: '#a78bfa', desc: `${gain >= 0 ? '+' : ''}${gain.toFixed(1)} vs best baseline` },
  ];
  $('#kpiGrid').innerHTML = kpis.map(k => `
    <div class="kpi">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value" style="color:${k.color}" title="${esc(k.value)}">${esc(k.value)}</div>
      <div class="kpi-desc">${esc(k.desc)}</div>
    </div>`).join('');
}

function renderScores() {
  const r = state.result;
  $('#scoreGrid').innerHTML = resultSubjects().map((s, i) => {
    const v = r.predicted[i];
    const b = scoreBand(v);
    return `
      <div class="score">
        <div class="score-name">${s.icon} ${esc(s.name)}</div>
        <div class="score-val">${v.toFixed(1)}%</div>
        <div class="meter"><span style="width:${v}%;background:${b.color}"></span></div>
        <div style="margin-top:8px"><span class="tag tag-${b.cls}">${b.label}</span></div>
      </div>`;
  }).join('');
}

function renderRiskTable() {
  const r = state.result;
  const rows = resultSubjects().map((s, i) => {
    const lbl = r.risk.label[i];
    const p = r.risk.priority[i];
    return `<tr>
      <td>${s.icon} ${esc(s.name)}</td>
      <td class="num-cell">${r.confidence[i].toFixed(1)}</td>
      <td class="num-cell">${r.predicted[i].toFixed(1)}%</td>
      <td><span class="prio-bar"><i><b style="width:${p * 10}%"></b></i><span class="num-cell">${p.toFixed(2)}</span></span></td>
      <td><span class="tag tag-${LABEL_CLASS[lbl] || 'warn'}">${esc(lbl)}</span></td>
    </tr>`;
  }).join('');
  $('#riskTable').innerHTML = `
    <thead><tr><th>Subject</th><th>Conf.</th><th>Predicted</th><th>Priority</th><th>Diagnosis</th></tr></thead>
    <tbody>${rows}</tbody>`;
}

function renderExplanations() {
  const r = state.result;
  $('#explainList').innerHTML = resultSubjects().map((s, i) => `
    <div class="explain" style="--c:${s.color}">
      <div class="explain-head">
        <span>${s.icon} ${esc(s.name)}</span>
        <span class="tag tag-${LABEL_CLASS[r.risk.label[i]] || 'warn'}">${esc(r.risk.label[i])}</span>
      </div>
      ${esc(r.risk.explanation[i])}
    </div>`).join('');
}

function renderDeltas() {
  const r = state.result;
  const deltas = r.allocation.hours.map((h, i) => h - r.input_hours[i]);
  const maxAbs = Math.max(1, ...deltas.map(Math.abs));
  $('#deltaList').innerHTML = resultSubjects().map((s, i) => {
    const d = deltas[i];
    const w = (Math.abs(d) / maxAbs) * 50;
    const bar = d >= 0
      ? `left:50%;width:${w}%;background:var(--good)`
      : `left:${50 - w}%;width:${w}%;background:var(--bad)`;
    return `
      <div class="delta">
        <span class="delta-name">${s.icon} ${esc(s.name)}</span>
        <span class="delta-track"><span style="${bar}"></span></span>
        <span class="delta-val ${d >= 0 ? 'up' : 'down'}">${d >= 0 ? '+' : ''}${d.toFixed(1)}h</span>
      </div>`;
  }).join('') + `<div class="delta-hint">${r.input_hours.map(h => h.toFixed(0)).join(' · ')} h → ${r.allocation.hours.map(h => h.toFixed(1)).join(' · ')} h</div>`;
}

function renderBaselines() {
  const a = state.result.allocation;
  const tiles = [
    { name: '🏆 Genetic algorithm', val: a.fitness, best: true },
    { name: 'Equal split', val: a.baselines.equal },
    { name: 'Proportional', val: a.baselines.proportional },
    { name: 'Greedy', val: a.baselines.greedy },
  ];
  $('#baselineGrid').innerHTML = tiles.map(t => {
    const diff = a.fitness - t.val;
    const sub = t.best
      ? '<span class="baseline-delta muted">optimised search</span>'
      : `<span class="baseline-delta ${diff >= 0 ? 'up' : 'down'}">GA ${diff >= 0 ? '+' : ''}${diff.toFixed(1)} pts</span>`;
    return `
      <div class="baseline${t.best ? ' best' : ''}">
        <div class="kpi-label">${t.name}</div>
        <div class="baseline-val">${t.val.toFixed(1)}</div>
        ${sub}
      </div>`;
  }).join('');
}

/* ---------- Timetable ---------- */
function toMin(hhmm) {
  const [h, m] = String(hhmm).split(':').map(Number);
  return h * 60 + (m || 0);
}

function layoutLanes(items) {
  // Assign overlapping blocks within a day to side-by-side lanes.
  const sorted = [...items].sort((a, b) => a.s - b.s);
  const laneEnds = [];
  for (const it of sorted) {
    let lane = laneEnds.findIndex(end => end <= it.s);
    if (lane === -1) { lane = laneEnds.length; laneEnds.push(0); }
    laneEnds[lane] = it.e;
    it.lane = lane;
  }
  for (const it of sorted) it.lanes = laneEnds.length;
  return sorted;
}

function renderWeek() {
  const sched = state.result.schedule;
  const colorOf = Object.fromEntries(resultSubjects().map(s => [s.name, s]));
  const items = sched.map(b => ({ ...b, s: toMin(b.start), e: toMin(b.end) }));

  const startH = Math.min(7, ...items.map(b => Math.floor(b.s / 60)));
  const endH = Math.max(21, ...items.map(b => Math.ceil(b.e / 60)));
  const hourH = 44;
  const height = (endH - startH) * hourH;
  const y = mins => ((mins - startH * 60) / 60) * hourH;

  const grid = $('#weekGrid');
  grid.innerHTML = '';
  grid.style.setProperty('--hour-h', `${hourH}px`);

  // Header row
  grid.append(el('div', { class: 'week-head' }));
  const dateOf = {};
  for (const b of items) if (/^\d{4}-\d{2}-\d{2}$/.test(b.date || '')) dateOf[b.day] = b.date;
  const todayName = DAYS[(new Date().getDay() + 6) % 7];
  for (const d of DAYS) {
    grid.append(el('div', { class: 'week-head' }, `${d.slice(0, 3)}<small>${dateOf[d] ? esc(dateOf[d].slice(5)) : '&nbsp;'}</small>`));
  }

  // Time gutter
  const times = el('div', { class: 'week-times', style: { height: `${height}px` } });
  for (let h = startH + 1; h < endH; h++) {
    times.append(el('span', { style: { top: `${(h - startH) * hourH}px` } }, `${String(h).padStart(2, '0')}:00`));
  }
  grid.append(times);

  // Day columns
  const peak = state.result.peak_window || '09:00-11:00';
  const [pS, pE] = peak.split('-').map(toMin);
  for (const d of DAYS) {
    const col = el('div', { class: `week-col${d === todayName ? ' today' : ''}`, style: { height: `${height}px` } });
    col.append(el('div', { class: 'peak', title: 'Cognitive peak window', style: { top: `${y(pS)}px`, height: `${y(pE) - y(pS)}px` } }));
    for (const b of layoutLanes(items.filter(it => it.day === d))) {
      const meta = colorOf[b.subject] || { color: '#6366f1', icon: '📚' };
      const width = 100 / b.lanes;
      col.append(el('div', {
        class: 'block',
        title: `${b.subject} · ${b.start}–${b.end} · ${b.reason || ''}`,
        style: {
          '--c': meta.color,
          top: `${y(b.s) + 1}px`,
          height: `${Math.max(22, y(b.e) - y(b.s) - 2)}px`,
          left: `calc(${b.lane * width}% + 3px)`,
          right: 'auto',
          width: `calc(${width}% - 6px)`,
        },
      }, `<b>${meta.icon} ${esc(b.subject)}</b><small>${esc(b.start)}–${esc(b.end)}</small>`));
    }
    grid.append(col);
  }

  const isStub = sched.every(b => (b.reason || '').toLowerCase().includes('stub'));
  $('#schedNote').textContent = isStub
    ? 'Stage 4 is still a stub — blocks are placeholders until the scheduler lands. Shaded band = 09:00–11:00 peak window.'
    : 'Blocks placed around the 09:00–11:00 peak window (shaded).';
}

/* ---------- Raw JSON ---------- */
function highlightJson(json) {
  return esc(json).replace(
    /(&quot;(?:\\.|[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/gi,
    (m, str, colon, num) => {
      if (str) return colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`;
      return `<span class="n">${num}</span>`;
    });
}

function renderRaw() {
  const r = state.result;
  const json = JSON.stringify(r, null, 2);
  $('#rawJson').innerHTML = highlightJson(json);
  const source = state.mode === 'demo'
    ? `${state.persona} · ${r.timings.total_ms} ms`
    : `Week of ${fmtWeek($('#weekInput').value)}`;
  $('#rawMeta').textContent = `${source} · ${(json.length / 1024).toFixed(1)} KB`;
}

/* ---------- Plotly charts ---------- */
function baseLayout(extra = {}) {
  const text = cssVar('--text-2');
  const grid = cssVar('--chart-grid');
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Plus Jakarta Sans, sans-serif', color: text, size: 12 },
    margin: { t: 20, r: 16, b: 40, l: 50 },
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid },
    hoverlabel: { bgcolor: cssVar('--surface-solid'), bordercolor: cssVar('--border-strong'), font: { color: cssVar('--text') } },
    legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right' },
    ...extra,
  };
}

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function hline(y, color, label, dash = 'dash') {
  return {
    shape: { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: y, y1: y, line: { color, width: 1.5, dash } },
    annotation: { xref: 'paper', x: 0, y, xanchor: 'left', yanchor: 'bottom', text: label, showarrow: false, font: { color, size: 11 } },
  };
}

function renderCharts() {
  if (!window.Plotly || !state.result) return;
  const r = state.result;
  const subj = resultSubjects();
  const names = subj.map(s => s.name);
  const colors = subj.map(s => s.color);

  // Stage 1: predicted marks
  const pass = hline(50, '#f59e0b', 'Pass 50%');
  const dist = hline(75, '#10b981', 'Distinction 75%');
  Plotly.react('chartPred', [{
    type: 'bar', x: names, y: r.predicted,
    marker: { color: colors, line: { color: 'rgba(255,255,255,0.15)', width: 1 } },
    text: r.predicted.map(v => `${v.toFixed(1)}%`), textposition: 'outside', cliponaxis: false,
    hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>',
  }], baseLayout({
    yaxis: { ...baseLayout().yaxis, title: 'Predicted mark (%)', range: [0, 105] },
    shapes: [pass.shape, dist.shape], annotations: [pass.annotation, dist.annotation],
    bargap: 0.45,
  }), PLOT_CONFIG);

  // Stage 2: calibration bubble
  Plotly.react('chartBubble', [{
    type: 'scatter', mode: 'markers+text',
    x: r.confidence, y: r.predicted, text: names, textposition: 'top center',
    marker: {
      size: r.risk.priority.map(p => 18 + p * 4.5), color: colors, opacity: 0.85,
      line: { color: 'rgba(255,255,255,0.7)', width: 1.5 },
    },
    customdata: r.risk.priority.map((p, i) => [p, r.risk.label[i]]),
    hovertemplate: '<b>%{text}</b><br>Confidence %{x}<br>Predicted %{y:.1f}%<br>Priority %{customdata[0]:.2f} · %{customdata[1]}<extra></extra>',
  }, {
    // Perfect-calibration reference: confidence c ↔ mark c*10
    type: 'scatter', mode: 'lines', x: [0, 10.5], y: [0, 105], hoverinfo: 'skip',
    line: { color: cssVar('--faint'), dash: 'dot', width: 1 }, name: 'Calibrated',
  }], baseLayout({
    showlegend: false,
    xaxis: { ...baseLayout().xaxis, title: 'Self-rated confidence', range: [0, 10.8] },
    yaxis: { ...baseLayout().yaxis, title: 'Predicted mark (%)', range: [0, 105] },
  }), PLOT_CONFIG);

  // Stage 3: allocation comparison
  Plotly.react('chartAlloc', [{
    type: 'bar', name: 'Your plan', x: names, y: r.input_hours,
    marker: { color: cssVar('--border-strong') },
    hovertemplate: '%{x}: %{y:.1f} h<extra>Your plan</extra>',
  }, {
    type: 'bar', name: 'Optimised', x: names, y: r.allocation.hours,
    marker: { color: '#6366f1' },
    hovertemplate: '%{x}: %{y:.1f} h<extra>Optimised</extra>',
  }], baseLayout({
    barmode: 'group', bargap: 0.3,
    yaxis: { ...baseLayout().yaxis, title: 'Hours / week' },
  }), PLOT_CONFIG);

  // Stage 3: GA convergence (baselines as flat reference traces with a legend)
  const b = r.allocation.baselines;
  const gens = r.allocation.history.map((_, i) => i);
  const ref = (name, y, color, dash) => ({
    type: 'scatter', mode: 'lines', name: `${name} ${y.toFixed(1)}`,
    x: [0, gens.length - 1], y: [y, y], line: { color, width: 1.5, dash },
    hovertemplate: `${name}: %{y:.2f}<extra></extra>`,
  });
  Plotly.react('chartConv', [
    ref('Equal', b.equal, '#f59e0b', 'dash'),
    ref('Proportional', b.proportional, '#f43f5e', 'dot'),
    ref('Greedy', b.greedy, '#10b981', 'dashdot'),
    {
      type: 'scatter', mode: 'lines', name: 'GA best',
      x: gens, y: r.allocation.history,
      line: { color: '#38bdf8', width: 2.5, shape: 'hv' },
      hovertemplate: 'Gen %{x}<br>Fitness %{y:.2f}<extra></extra>',
    },
  ], baseLayout({
    xaxis: { ...baseLayout().xaxis, title: 'Generation' },
    yaxis: { ...baseLayout().yaxis, title: 'Fitness' },
  }), PLOT_CONFIG);

  // Stage 4: time share donut
  Plotly.react('chartPie', [{
    type: 'pie', hole: 0.62, labels: names, values: r.allocation.hours,
    marker: { colors, line: { color: cssVar('--surface-solid'), width: 2 } },
    textinfo: 'percent', sort: false,
    hovertemplate: '%{label}<br>%{value:.1f} h (%{percent})<extra></extra>',
  }], baseLayout({
    margin: { t: 10, r: 10, b: 10, l: 10 },
    legend: { orientation: 'h', y: -0.1, x: 0.5, xanchor: 'center' },
    annotations: [{
      text: `<b>${r.budget.toFixed(0)} h</b><br><span style="font-size:11px">per week</span>`,
      showarrow: false, font: { size: 18, color: cssVar('--text') },
    }],
  }), PLOT_CONFIG);

  // Stage 4: hours per day, stacked by subject
  const perDay = names.map(n => DAYS.map(d => r.schedule
    .filter(blk => blk.day === d && blk.subject === n)
    .reduce((sum, blk) => sum + blk.minutes / 60, 0)));
  Plotly.react('chartDaily', names.map((n, i) => ({
    type: 'bar', name: n, x: DAYS.map(d => d.slice(0, 3)), y: perDay[i],
    marker: { color: colors[i] },
    hovertemplate: `${n}: %{y:.1f} h<extra></extra>`,
  })), baseLayout({
    barmode: 'stack', bargap: 0.35,
    yaxis: { ...baseLayout().yaxis, title: 'Hours' },
  }), PLOT_CONFIG);
}

/* =========================================================================
   Tabs (WAI-ARIA tabs pattern)
   ========================================================================= */
function initTabs() {
  const tabs = [...document.querySelectorAll('.tabs [role="tab"]')];
  const select = tab => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    }
    state.activeTab = tab.id.replace('tab-', '');
    // Plotly needs a resize once its container becomes visible
    requestAnimationFrame(() => {
      document.querySelectorAll(`#${tab.getAttribute('aria-controls')} .js-plotly-plot`)
        .forEach(p => Plotly.Plots.resize(p));
    });
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t));
    t.addEventListener('keydown', e => {
      const dir = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!dir) return;
      const next = tabs[(i + dir + tabs.length) % tabs.length];
      next.focus();
      select(next);
    });
  });
}

/* =========================================================================
   Accounts & modes
   ========================================================================= */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('[data-show]').forEach(n => { n.hidden = n.dataset.show !== mode; });
  $('#authView').hidden = true;
  $('#shell').hidden = false;

  const user = mode === 'user';
  $('#pageTitle').textContent = user
    ? `Hi ${state.user.name.split(' ')[0]}, how did this week go?`
    : 'Smart Learning & Performance Optimizer';
  $('#pageSub').textContent = user
    ? 'Log your hours and confidence. We predict your marks, flag blind spots and rebuild your study plan.'
    : 'ML regression → Mamdani fuzzy risk → Genetic Algorithm reallocation → weekly schedule.';
  $('#inputTitle').textContent = user ? 'Weekly check-in' : 'Semester telemetry & self-assessment';
  $('#inputSub').textContent = user
    ? 'Hours you studied each subject this week, and how confident you feel about it.'
    : 'Weekly target hours and perceived confidence per subject.';
  $('.run-label').textContent = runLabel();
  if (user) $('#liveToggle').checked = false;
}

function showAuth(tab = 'login', message = '') {
  state.mode = null;
  state.user = null;
  state.result = null;
  if ($('#subjectDialog').open) $('#subjectDialog').close();
  $('#shell').hidden = true;
  $('#authView').hidden = false;
  setAuthTab(tab);
  $('#authError').textContent = message;
  $('#authError').hidden = !message;
  if (location.hash) history.replaceState(null, '', location.pathname + location.search);
  $('#authForm [name="email"]').focus();
}

function setAuthTab(tab) {
  const register = tab === 'register';
  $('#authTabLogin').setAttribute('aria-selected', String(!register));
  $('#authTabRegister').setAttribute('aria-selected', String(register));
  $('#nameField').hidden = !register;
  $('#pwHint').hidden = !register;
  $('#authForm [name="password"]').autocomplete = register ? 'new-password' : 'current-password';
  $('#authTitle').textContent = register ? 'Create your account' : 'Welcome back';
  $('.auth-submit-label').textContent = register ? 'Create account' : 'Sign in';
  $('#authError').hidden = true;
}

async function submitAuth(e) {
  e.preventDefault();
  const form = e.target;
  const register = $('#authTabRegister').getAttribute('aria-selected') === 'true';
  const body = { email: form.email.value.trim(), password: form.password.value };
  if (register) body.name = form.elements.name.value.trim();

  const problem = !body.email ? 'Enter your email.'
    : !body.password ? 'Enter your password.'
    : register && !body.name ? 'Enter your name.'
    : register && body.password.length < 8 ? 'Password must be at least 8 characters.'
    : '';
  if (problem) {
    $('#authError').textContent = problem;
    $('#authError').hidden = false;
    return;
  }

  const btn = $('#authSubmit');
  btn.classList.add('is-loading');
  btn.disabled = true;
  try {
    const res = await api(register ? '/api/auth/register' : '/api/auth/login', body);
    state.user = res.user;
    form.reset();
    await enterUserMode();
    if (register) toast(`👋 Welcome, ${state.user.name.split(' ')[0]}!`);
  } catch (err) {
    $('#authError').textContent = err.message;
    $('#authError').hidden = false;
  } finally {
    btn.classList.remove('is-loading');
    btn.disabled = false;
  }
}

async function logout() {
  try { await api('/api/auth/logout', {}); } catch { /* nothing to undo */ }
  showAuth('login');
  toast('Signed out');
}

async function deleteAccount() {
  const ok = confirm(
    'Delete your account?\n\nThis permanently removes your profile, subjects, every weekly check-in and every plan. It cannot be undone.');
  if (!ok) return;
  try {
    await api('/api/me', undefined, 'DELETE');
    showAuth('register');
    toast('Your account and data were deleted');
  } catch (err) {
    showSubjectError(err.message);
  }
}

async function enterUserMode() {
  const [meta, subjects, checkins] = await Promise.all([
    api('/api/me/meta'), api('/api/subjects'), api('/api/checkins'),
  ]);
  state.userMeta = meta;
  state.subjects = subjects;
  state.checkins = checkins;
  state.viewing = null;
  if (location.hash) history.replaceState(null, '', location.pathname + location.search);

  setMode('user');
  renderAccount();
  renderSubjectList();
  renderHistory();

  const week = $('#weekInput');
  week.max = meta.current_week;
  week.value = meta.current_week;

  const current = checkins.find(c => c.week_start === meta.current_week);
  if (current) {
    await openCheckin(current.id);
  } else {
    // Start this week from the latest week's numbers rather than from scratch
    const latest = checkins[0] ? await api(`/api/checkins/${checkins[0].id}`) : null;
    setInputsFrom(latest ? entryMap(latest) : {});
    clearResults();
    refreshInputs();
  }
  if (!subjects.length) openSubjectDialog();
}

async function enterDemo() {
  if (!state.meta) {
    state.meta = await api('/api/meta');
    state.logins = state.meta.lms_logins;
  }
  setMode('demo');
  if (location.hash !== '#demo') history.replaceState(null, '', '#demo');
  selectPersona(state.persona || Object.keys(state.meta.personas)[0]);
}

function renderAccount() {
  $('#accountAvatar').textContent = (state.user.name.trim()[0] || '?').toUpperCase();
  $('#accountName').textContent = state.user.name;
  $('#accountEmail').textContent = state.user.email;
}

/* ---------- Inputs for the signed-in user ---------- */
const entryMap = ck => Object.fromEntries(ck.entries.map(e => [e.subject_id, { hours: e.hours, conf: e.confidence }]));
const currentInputMap = () => Object.fromEntries(
  state.subjects.map((s, i) => [s.id, { hours: state.hours[i], conf: state.conf[i] }]));

function setInputsFrom(map) {
  const [hMin, hMax] = state.userMeta.hours_range;
  const [cMin, cMax] = state.userMeta.conf_range;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  state.hours = state.subjects.map(s => clamp(map[s.id]?.hours ?? DEFAULT_HOURS, hMin, hMax));
  state.conf = state.subjects.map(s => clamp(map[s.id]?.conf ?? DEFAULT_CONF, cMin, cMax));
}

function refreshInputs() {
  const none = state.mode === 'user' && !state.subjects.length;
  $('#inputPanel').hidden = none;
  $('#noSubjects').hidden = !none;
  $('#runBtn').disabled = none;
  if (none) { $('#emptyState').hidden = true; return; }
  renderSubjects();
}

function examPill(iso) {
  const d = daysUntil(iso);
  if (d < 0) return '';
  const label = d === 0 ? 'Exam today' : d === 1 ? 'Exam tomorrow' : `Exam in ${d}d`;
  return `<span class="pill" title="Exam ${esc(fmtWeek(iso))}">${label}</span>`;
}

async function saveCheckin() {
  const entries = state.subjects.map((s, i) => ({ subject_id: s.id, hours: state.hours[i], confidence: state.conf[i] }));
  const ck = await api('/api/checkins', { week_start: $('#weekInput').value || null, entries });
  state.result = ck.plan.result;
  state.viewing = ck.id;
  $('#weekInput').value = ck.week_start;
  state.checkins = await api('/api/checkins');
  renderHistory();
}

async function openCheckin(id) {
  const ck = await api(`/api/checkins/${id}`);
  state.viewing = ck.id;
  $('#weekInput').value = ck.week_start;
  setInputsFrom(entryMap(ck));
  refreshInputs();
  state.result = ck.plan.result;
  setStepper(5);
  renderResults();
  renderHistory();
  closeSidebar();
}

async function onWeekChanged() {
  const input = $('#weekInput');
  if (!input.value) return;
  let week = mondayOf(input.value);
  if (week > state.userMeta.current_week) week = state.userMeta.current_week;
  input.value = week;
  const existing = state.checkins.find(c => c.week_start === week);
  if (existing) {
    await openCheckin(existing.id);
  } else {
    state.viewing = null;
    clearResults();
    renderHistory();
  }
}

/* ---------- Sidebar: subjects & history ---------- */
function renderSubjectList() {
  const list = $('#subjectList');
  if (!state.subjects.length) {
    list.innerHTML = '<li class="empty-line">No subjects yet</li>';
    return;
  }
  list.innerHTML = state.subjects.map(s => {
    const d = s.exam_date ? daysUntil(s.exam_date) : null;
    const meta = d !== null && d >= 0 ? `exam ${d}d` : DIFFICULTY[s.difficulty];
    return `<li style="--c:${s.color}"><span class="swatch"></span>${esc(s.name)}<span class="sub-meta">${esc(meta)}</span></li>`;
  }).join('');
}

function renderHistory() {
  const list = $('#historyList');
  if (!state.checkins.length) {
    list.innerHTML = '<li class="empty-line">Your saved weeks will appear here.</li>';
    return;
  }
  list.innerHTML = '';
  state.checkins.forEach((c, i) => {
    const prev = state.checkins[i + 1];
    const delta = prev ? c.avg_predicted - prev.avg_predicted : null;
    const trend = delta === null || Math.abs(delta) < 0.05 ? ''
      : `<span class="${delta > 0 ? 'up' : 'down'}"> ${delta > 0 ? '▲' : '▼'}${Math.abs(delta).toFixed(1)}</span>`;
    const label = c.week_start === state.userMeta.current_week ? 'This week' : `Week of ${fmtWeek(c.week_start)}`;
    const btn = el('button', {
      type: 'button',
      'aria-current': String(c.id === state.viewing),
      title: 'Average predicted mark across subjects',
      onclick: () => openCheckin(c.id).catch(err => toast(err.message)),
    }, `<span class="h-week">${esc(label)}</span>
        <span class="h-score">${c.avg_predicted.toFixed(1)}%${trend}</span>
        <span class="h-meta">${c.budget.toFixed(0)} h · focus: ${esc(c.top_priority)}</span>`);
    const li = el('li');
    li.append(btn);
    list.append(li);
  });
}

/* ---------- Subject manager dialog ---------- */
function openSubjectDialog() {
  renderSubjectRows();
  $('#subjectError').hidden = true;
  $('#subjectDialog').showModal();
  closeSidebar();
  $('#addSubjectForm [name="name"]').focus();
}

function showSubjectError(msg) {
  $('#subjectError').textContent = msg;
  $('#subjectError').hidden = false;
}

async function reloadSubjects() {
  const keep = currentInputMap();
  state.subjects = await api('/api/subjects');
  setInputsFrom(keep);
  renderSubjectRows();
  renderSubjectList();
  refreshInputs();
  if (!state.result) clearResults();
}

async function mutateSubject(fn) {
  $('#subjectError').hidden = true;
  try {
    await fn();
    await reloadSubjects();
    return true;
  } catch (err) {
    showSubjectError(err.message);
    renderSubjectRows();  // put back the last saved values
    return false;
  }
}

function renderSubjectRows() {
  const body = $('#subjectRows');
  body.innerHTML = '';
  if (!state.subjects.length) {
    body.innerHTML = '<tr><td colspan="4" class="muted">No subjects yet. Add your first one below.</td></tr>';
  }
  const atLimit = state.subjects.length >= state.userMeta.max_subjects;
  $('#addSubjectForm button').disabled = atLimit;
  $('#addSubjectForm [name="name"]').placeholder = atLimit
    ? `Limit of ${state.userMeta.max_subjects} subjects reached` : 'e.g. Operating Systems';

  for (const s of state.subjects) {
    const row = el('tr', { style: { '--c': s.color } }, `
      <td><span class="name-cell"><span class="swatch"></span><input type="text" value="${esc(s.name)}" maxlength="60" aria-label="Subject name"></span></td>
      <td><select aria-label="Difficulty of ${esc(s.name)}">
        ${Object.entries(DIFFICULTY).map(([v, l]) => `<option value="${v}"${+v === s.difficulty ? ' selected' : ''}>${l}</option>`).join('')}
      </select></td>
      <td><input type="date" value="${s.exam_date || ''}" aria-label="Exam date for ${esc(s.name)}"></td>
      <td><button type="button" class="icon-btn" aria-label="Remove ${esc(s.name)}">Remove</button></td>`);
    const [nameIn, dateIn] = row.querySelectorAll('input');
    const patch = changes => mutateSubject(() => api(`/api/subjects/${s.id}`, changes, 'PATCH'));
    nameIn.addEventListener('change', () => {
      if (nameIn.value.trim() && nameIn.value.trim() !== s.name) patch({ name: nameIn.value });
      else nameIn.value = s.name;
    });
    $('select', row).addEventListener('change', e => patch({ difficulty: +e.target.value }));
    dateIn.addEventListener('change', () => patch({ exam_date: dateIn.value || null }));
    $('button', row).addEventListener('click', () => {
      if (confirm(`Remove ${s.name}? Past check-ins that include it are kept.`)) {
        mutateSubject(() => api(`/api/subjects/${s.id}`, undefined, 'DELETE'));
      }
    });
    body.append(row);
  }
}

async function addSubject(e) {
  e.preventDefault();
  const form = e.target;
  const name = form.elements.name.value.trim();
  if (!name) { form.elements.name.focus(); return; }
  const ok = await mutateSubject(() => api('/api/subjects', {
    name,
    difficulty: +form.elements.difficulty.value,
    exam_date: form.elements.exam_date.value || null,
  }));
  if (ok) {
    form.reset();
    form.elements.name.focus();
  }
}

/* =========================================================================
   Boot
   ========================================================================= */
async function boot() {
  initTheme();
  initTabs();

  $('#runBtn').addEventListener('click', runPipeline);
  $('#syncBtn').addEventListener('click', syncLms);
  $('#resetBtn').addEventListener('click', () => selectPersona(state.persona));
  $('#menuBtn').addEventListener('click', openSidebar);
  $('#scrim').addEventListener('click', closeSidebar);
  $('#liveToggle').addEventListener('change', e => { if (e.target.checked) runPipeline(); });
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !$('#shell').hidden && !$('#subjectDialog').open) runPipeline();
    if (e.key === 'Escape') closeSidebar();
  });

  // Accounts
  $('#authTabLogin').addEventListener('click', () => setAuthTab('login'));
  $('#authTabRegister').addEventListener('click', () => setAuthTab('register'));
  $('#authForm').addEventListener('submit', submitAuth);
  $('#demoLink').addEventListener('click', e => {
    e.preventDefault();
    enterDemo().catch(err => { $('#authError').textContent = err.message; $('#authError').hidden = false; });
  });
  $('#exitDemo').addEventListener('click', e => { e.preventDefault(); showAuth('register'); });
  $('#logoutBtn').addEventListener('click', logout);
  $('#weekInput').addEventListener('change', () => onWeekChanged().catch(err => toast(err.message)));

  // Subject manager
  $('#manageBtn').addEventListener('click', openSubjectDialog);
  $('#addFirstBtn').addEventListener('click', openSubjectDialog);
  $('#dialogClose').addEventListener('click', () => $('#subjectDialog').close());
  $('#dialogDone').addEventListener('click', () => $('#subjectDialog').close());
  $('#addSubjectForm').addEventListener('submit', addSubject);
  $('#deleteAccountBtn').addEventListener('click', deleteAccount);

  $('#copyBtn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(state.result, null, 2));
      toast('📋 JSON copied');
    } catch { toast('Clipboard unavailable'); }
  });
  $('#downloadBtn').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' });
    const a = el('a', { href: URL.createObjectURL(blob), download: 'pipeline_result.json' });
    a.click();
    URL.revokeObjectURL(a.href);
  });

  try {
    try {
      state.user = await api('/api/me');
    } catch (err) {
      if (err.status !== 401) throw err;
    }
    $('#apiStatus').textContent = '● API online';

    if (state.user) await enterUserMode();
    else if (location.hash === '#demo') await enterDemo();
    else { showAuth('login'); return; }

    // Deep links for demos: ?run=1&tab=risk
    const params = new URLSearchParams(location.search);
    const tab = document.getElementById(`tab-${params.get('tab')}`);
    if (tab) tab.click();
    if (params.has('run') && state.mode === 'demo') runPipeline();
  } catch (e) {
    showAuth('login', `Could not reach the API: ${e.message}. Start it with "uvicorn app.api:app --reload" from the project root.`);
  }
}

document.addEventListener('DOMContentLoaded', boot);
