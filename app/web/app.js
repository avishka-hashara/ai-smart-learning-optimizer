'use strict';

/* =========================================================================
   State
   ========================================================================= */
const state = {
  meta: null,          // /api/meta payload
  persona: null,       // active persona name
  hours: [],
  conf: [],
  logins: [],
  result: null,        // last /api/run payload
  running: false,
  activeTab: 'pred',
};

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

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
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
  const [hMin, hMax] = state.meta.hours_range;
  const [cMin, cMax] = state.meta.conf_range;
  grid.innerHTML = '';

  state.meta.subjects.forEach((s, i) => {
    const card = el('div', { class: 'subject', style: { '--c': s.color } }, `
      <div class="subject-top">
        <span class="subject-name">${s.icon} ${esc(s.name)}</span>
        <span class="pill" title="LMS logins this week">📡 ${state.logins[i]}</span>
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
  state.meta.subjects.forEach((s, i) => {
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

async function runPipeline() {
  if (state.running) return;
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
    state.result = await api('/api/run', { hours: state.hours, confidence: state.conf });
    clearInterval(ticker);
    setStepper(5);
    renderResults();
    if (!$('#liveToggle').checked) toast(`✨ Pipeline finished in ${state.result.timings.total_ms} ms`);
  } catch (e) {
    clearInterval(ticker);
    setStepper(0);
    $('#errorBox').textContent = e.message;
    $('#errorBox').hidden = false;
  } finally {
    state.running = false;
    btn.classList.remove('is-loading');
    btn.disabled = false;
    $('.run-label', btn).textContent = 'Run pipeline';
  }
}

function clearResults() {
  state.result = null;
  $('#results').hidden = true;
  $('#emptyState').hidden = false;
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
  const subj = state.meta.subjects;
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
  $('#scoreGrid').innerHTML = state.meta.subjects.map((s, i) => {
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
  const rows = state.meta.subjects.map((s, i) => {
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
  $('#explainList').innerHTML = state.meta.subjects.map((s, i) => `
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
  $('#deltaList').innerHTML = state.meta.subjects.map((s, i) => {
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
  const colorOf = Object.fromEntries(state.meta.subjects.map(s => [s.name, s]));
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
  $('#rawMeta').textContent = `${state.persona} · ${r.timings.total_ms} ms · ${(json.length / 1024).toFixed(1)} KB`;
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
  const subj = state.meta.subjects;
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
  const tabs = [...document.querySelectorAll('[role="tab"]')];
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
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runPipeline();
    if (e.key === 'Escape') closeSidebar();
  });

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
    state.meta = await api('/api/meta');
    state.logins = state.meta.lms_logins;
    $('#apiStatus').textContent = '● API online';
    selectPersona(Object.keys(state.meta.personas)[0]);

    // Deep links for demos: ?run=1&tab=risk
    const params = new URLSearchParams(location.search);
    const tab = document.getElementById(`tab-${params.get('tab')}`);
    if (tab) tab.click();
    if (params.has('run')) runPipeline();
  } catch (e) {
    $('#apiStatus').textContent = '● API offline';
    $('#errorBox').textContent = `Could not reach the API: ${e.message}. Start it with "uvicorn app.api:app --reload" from the project root.`;
    $('#errorBox').hidden = false;
  }
}

document.addEventListener('DOMContentLoaded', boot);
