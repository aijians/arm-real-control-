'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const axes = ['upper', 'lower'];
  const labels = {upper:'大臂', lower:'小臂'};
  const initialized = new Set();
  let token = '', state = null, busy = false, polling = false, framesLoaded = false;
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => number(value) ? value.toFixed(2) : '—';
  try { token = sessionStorage.getItem('pid_tuner_token') || ''; } catch (_) {}
  const supplied = new URLSearchParams(location.hash.slice(1)).get('token');
  if (supplied) { token = supplied; history.replaceState(null,'',location.pathname); }
  function remember() { try { sessionStorage.setItem('pid_tuner_token', token); } catch (_) {} }
  remember();
  function message(text, error = false) {
    $('dual-message').textContent = text;
    $('dual-message').className = `message ${error ? '' : 'success'}`;
    $('dual-message').hidden = false;
  }
  async function api(path, data) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(path,{method:data === undefined ? 'GET':'POST',
        headers:{Authorization:`Bearer ${token}`,...(data === undefined ? {}:{'Content-Type':'application/json'})},
        body:data === undefined ? undefined:JSON.stringify(data),cache:'no-store',signal:controller.signal});
      const result = await response.json();
      if (!response.ok || result.ok === false) {
        if (response.status === 401) $('access').hidden = false;
        throw new Error(result.error || `请求失败 ${response.status}`);
      }
      return result;
    } finally { clearTimeout(timeout); }
  }
  function canMove(axis) {
    const s = state?.axes?.[axis];
    const m = s?.motion;
    const verified = !Object.values(s?.pid_status || {}).some(v => ['write_pending','write_unverified','rom_unverified','changed_on_device'].includes(v));
    return s?.connected && s.latest?.motor_state === 0 && !s.latest?.error_state && number(s.latest?.angle_deg)
      && s.last_feedback_age_s <= 1 && !m?.stop_required && verified
      && (!m?.active || (m.phase === 'holding' && m.trial_outcome === 'passed' && !m.hold_warning));
  }
  function render(next) {
    state = next;
    $('dual-mode').textContent = state.mode === 'demo' ? 'DEMO · 双轴演示' : '实机 · 双轴控制';
    $('dual-mode').className = `badge ${state.mode === 'demo' ? 'demo':'real'}`;
    $('dual-connect').disabled = busy || axes.every(a => state.axes[a].connected);
    $('dual-disconnect').disabled = busy || !axes.some(a => state.axes[a].connected);
    $('dual-move').disabled = busy || !axes.every(canMove);
    $('dual-current').disabled = busy || !axes.every(a => state.axes[a].connected && number(state.axes[a].latest?.angle_deg));
    for (const axis of axes) {
      const s = state.axes[axis], m = s.motion || {}, latest = s.latest || {};
      $(`${axis}-status`).textContent = m.stop_required ? '停止未确认' : m.active ? (m.phase === 'holding' ? '保持中':'移动中') : s.connected ? '已连接':'未连接';
      $(`${axis}-device`).textContent = `${s.connection.device} · ID ${s.connection.motor_id}`;
      $(`${axis}-angle`).textContent = fmt(latest.angle_deg);
      $(`${axis}-speed`).textContent = `${fmt(latest.speed_deg_s)} °/s`;
      $(`${axis}-limits`).textContent = `限位 ${s.limits.min_deg}° ～ ${s.limits.max_deg}° · 单次 ≤10°`;
      $(`${axis}-target`).min = s.limits.min_deg; $(`${axis}-target`).max = s.limits.max_deg;
      if (number(latest.angle_deg) && !initialized.has(axis)) {
        $(`${axis}-target`).value = latest.angle_deg.toFixed(2); initialized.add(axis);
      }
      $(`${axis}-target`).disabled = busy || (m.active && !(m.phase === 'holding' && m.trial_outcome === 'passed'));
      $(`${axis}-move-speed`).disabled = $(`${axis}-target`).disabled;
      const status = {pending:'验证中',passed:'保持验证通过',failed:'试转失败',cancelled:'试转已取消'}[m.trial_outcome] || '待机';
      $(`${axis}-result`).textContent = `${status}${number(m.target_deg) ? ` · 目标 ${fmt(m.target_deg)}°`:''}${s.last_error ? ` · ${s.last_error}`:''}`;
    }
    $('dual-group').textContent = state.group.stop_reason || ({preparing:'双轴检查与下发中',running:'双轴正在试转',holding:'双轴均已验证通过，持续保持',stopped:'双轴已停止',stop_unconfirmed:'双轴停止尚未全部确认'}[state.group.result] || '双轴待机');
    if (!framesLoaded) {
      framesLoaded = true;
      for (const axis of axes) $(`frame-${axis}`).src = `/axis/${axis}#token=${encodeURIComponent(token)}`;
    }
    $('access').hidden = true;
  }
  async function poll() {
    if (!token || polling) return;
    polling = true;
    try { render(await api('/api/dual/state')); }
    catch (error) { $('dual-move').disabled = true; message(`服务连接异常：${error.message}`, true); }
    finally { polling = false; }
  }
  async function action(path, data, text) {
    const stop = path === '/api/dual/stop';
    if (busy && !stop) return;
    if (!stop) busy = true;
    if (state) render(state);
    try { await api(path,data); message(text); }
    catch (error) { message(`操作未确认成功：${error.message}`,true); }
    finally { if (!stop) busy = false; await poll(); }
  }
  $('dual-connect').onclick = () => action('/api/dual/connect',{},'两轴连接完成；未自动使能或移动。');
  $('dual-disconnect').onclick = () => action('/api/dual/disconnect',{},'两轴已停止并断开。');
  $('dual-stop').onclick = () => action('/api/dual/stop',{},'两轴停止已确认；注意停止后机械臂可能回位。');
  $('dual-current').onclick = () => {
    for (const axis of axes) if (number(state?.axes[axis]?.latest?.angle_deg)) $(`${axis}-target`).value = state.axes[axis].latest.angle_deg.toFixed(2);
  };
  $('dual-move').onclick = () => {
    const targets = {};
    for (const axis of axes) {
      const target = Number($(`${axis}-target`).value), speed = Number($(`${axis}-move-speed`).value);
      const s = state?.axes?.[axis];
      if (!$(`${axis}-target`).value.trim() || !$(`${axis}-move-speed`).value.trim() || !number(target) || !number(speed) || speed <= 0 || speed > 60
          || !s || target < s.limits.min_deg || target > s.limits.max_deg || Math.abs(target - s.latest.angle_deg) > 10 + 1e-6) {
        message(`${labels[axis]}目标需在配置限位内、距离当前位置不超过10°，速度为0～60°/s（不含0）。`,true); return;
      }
      targets[axis] = {target_deg:target,speed_deg_s:speed};
    }
    $('dual-confirm-text').textContent = axes.map(a => `${labels[a]}：${targets[a].target_deg}°，${targets[a].speed_deg_s}°/s`).join('\n') + '\n两轴将并发下发，各自到位验证后持续保持。任一轴失败会停止双轴。确认行程已清空、机械臂已支撑，并留在页面观察。';
    const dialog = $('dual-confirm');
    dialog.returnValue = '';
    dialog.addEventListener('close',() => {
      if (dialog.returnValue === 'confirm') action('/api/dual/move',{targets},'双轴试转已下发；到位后分别验证并保持。');
    },{once:true});
    dialog.showModal();
  };
  for (const axis of axes) $(`tab-${axis}`).onclick = () => {
    for (const other of axes) {
      $(`frame-${other}`).hidden = other !== axis;
      $(`tab-${other}`).setAttribute('aria-selected',String(other === axis));
      $(`tab-${other}`).className = `button ${other === axis ? 'primary':'secondary'}`;
    }
  };
  $('access').onsubmit = event => {
    event.preventDefault(); token = $('access-token').value.trim(); remember();
    framesLoaded = false; poll();
  };
  window.addEventListener('beforeunload',event => {
    if (axes.some(a => state?.axes?.[a]?.motion?.active)) { event.preventDefault(); event.returnValue = ''; }
  });
  if (!token) $('access').hidden = false;
  poll(); setInterval(poll,500);
})();
