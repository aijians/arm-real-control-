'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const loops = ['angle', 'speed'];
  const gains = ['kp', 'ki', 'kd'];
  const drafts = {angle: {initialized: false, dirty: false}, speed: {initialized: false, dirty: false}};
  let state = null;
  let token = '';
  let serviceHealthy = false;
  let commandBusy = false;
  let pollPromise = null;
  let sampleSeq = 0;
  let samples = [];
  let lastLogSeq = 0;
  let seenSession = null;
  let localLogs = [];
  let serverLogs = [];
  let targetEdited = false;
  let targetInitialized = false;
  let sessionEpoch = Date.now();
  let lastServerError = '';
  let lastDiagnosticsSignature = '';
  let lastReceivedAt = 0;
  let consecutiveErrors = 0;
  let confirmResolve = null;
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits = 2) => number(value) ? value.toFixed(digits) : '—';
  const nameOf = loop => loop === 'angle' ? '角度环' : '速度环';
  const axisLabel = () => state?.connection?.axis_label || '所选关节';
  const vendorSettings = () => state?.save_scope === 'all_settings';
  const draftReference = loop => vendorSettings() ? state?.ram_ack_values?.[loop] || state?.params?.[loop] : state?.params?.[loop];
  const pidText = values => gains.map(gain => `${gain.toUpperCase()}=${values?.[gain] ?? '—'}`).join('，');
  const pidFromRaw = hex => {
    const bytes = String(hex || '').trim().split(/\s+/);
    if (bytes.length !== 6 || bytes.some(byte => !/^[0-9a-f]{2}$/i.test(byte))) return null;
    return Object.fromEntries(gains.map((gain, index) => [gain, parseInt(bytes[index * 2], 16) + 256 * parseInt(bytes[index * 2 + 1], 16)]));
  };

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  function rememberToken(value) {
    token = value.trim();
    try { sessionStorage.setItem('pid_tuner_token', token); } catch (_) { /* session-only access still works */ }
  }

  function showMessage(message, type = 'error') {
    $('message-text').textContent = message;
    $('message').className = `message ${type}`;
    $('message').hidden = false;
  }

  function localLog(message, level = 'info') {
    localLogs.push({time: new Date().toISOString(), level, message, local: true});
    localLogs = localLogs.slice(-80);
    renderLogs();
  }

  function askToken(error = '') {
    $('token-error').textContent = error;
    if (!$('token-dialog').open) {
      $('token-dialog').showModal();
      setTimeout(() => $('token-input').focus(), 0);
    }
  }

  async function request(path, data, options = {}) {
    let response;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || 12000);
    try {
      response = await fetch(path, {
        method: data === undefined ? 'GET' : 'POST',
        headers: {Authorization: `Bearer ${token}`, ...(data === undefined ? {} : {'Content-Type': 'application/json'})},
        body: data === undefined ? undefined : JSON.stringify(data),
        cache: 'no-store', signal: controller.signal
      });
    } catch (error) {
      throw new ApiError(error.name === 'AbortError' ? '请求超时，请检查开发板服务与串口。' : '无法访问开发板服务，请检查网络或服务是否仍在运行。', 0);
    } finally {
      clearTimeout(timeout);
    }
    if (options.blob && response.ok) return response.blob();
    let result;
    try { result = await response.json(); } catch (_) { throw new ApiError(`服务返回无效响应（HTTP ${response.status}）。`, response.status); }
    if (!response.ok || result.ok === false) {
      if (response.status === 401 || response.status === 403) askToken('访问令牌无效或已过期，请重新输入。');
      throw new ApiError(result.error || `请求失败（HTTP ${response.status}）`, response.status);
    }
    return result;
  }

  function confirmAction(title, description, confirmText = '确认') {
    if (confirmResolve) return Promise.resolve(false);
    $('confirm-title').textContent = title;
    $('confirm-description').textContent = description;
    $('confirm-submit').textContent = confirmText;
    $('confirm-dialog').returnValue = '';
    $('confirm-dialog').showModal();
    return new Promise(resolve => { confirmResolve = resolve; });
  }

  $('confirm-dialog').addEventListener('close', () => {
    const resolve = confirmResolve;
    confirmResolve = null;
    if (resolve) resolve($('confirm-dialog').returnValue === 'confirm');
  });

  function formValues(loop) {
    const result = {};
    for (const gain of gains) {
      const input = $(`${loop}-${gain}`);
      const value = Number(input.value);
      if (input.value.trim() === '' || !Number.isInteger(value) || value < 0 || value > 65535) {
        input.focus();
        throw new Error(`${nameOf(loop)} ${gain.toUpperCase()} 必须为 0–65535 的整数。`);
      }
      result[gain] = value;
    }
    return result;
  }

  function markDraft(loop) {
    const current = draftReference(loop);
    const draft = drafts[loop];
    draft.dirty = gains.some(gain => $(`${loop}-${gain}`).value.trim() === '' || Number($(`${loop}-${gain}`).value) !== current?.[gain]);
    for (const gain of gains) {
      const input = $(`${loop}-${gain}`);
      input.classList.toggle('dirty', input.value.trim() === '' || Number(input.value) !== current?.[gain]);
    }
    const text = $(`${loop}-draft`);
    text.classList.toggle('dirty', draft.dirty);
    const uncertain = ['write_unverified', 'rom_unverified'].includes(state?.pid_status?.[loop]);
    text.textContent = uncertain ? '写入结果未确认 · 不能据此判断运行参数' : !draft.initialized ? '读取后可编辑' : draft.dirty ? '● 表单已修改 · 尚未下发此目标' : vendorSettings() && state?.ram_ack_values?.[loop] ? '与本会话已应答目标一致 · 运行值未读回' : vendorSettings() ? '与整组设置读回一致 · 不代表运行 RAM' : '与当前读回值一致';
    refreshControls();
  }

  function populateParams() {
    for (const loop of loops) {
      const params = state?.params?.[loop];
      const baseline = state?.baseline?.[loop];
      if (params && gains.every(gain => number(params[gain]))) {
        // Keep the acknowledged target separate from the vendor settings readback.
        // Polling never replaces an edited form with older settings values.
        if (!drafts[loop].initialized || !drafts[loop].dirty) {
          const reference = draftReference(loop);
          for (const gain of gains) $(`${loop}-${gain}`).value = reference[gain];
          drafts[loop].initialized = true;
        }
        for (const gain of gains) $(`${loop}-${gain}-read`).textContent = params[gain];
        for (const label of document.querySelectorAll(`[data-readback-loop="${loop}"]`)) label.textContent = vendorSettings() ? '整组读回' : '参数读回';
        markDraft(loop);
      }
      for (const gain of gains) $(`${loop}-${gain}-backup`).textContent = baseline?.[gain] ?? '—';
      const status = state?.pid_status?.[loop];
      const statuses = {
        read: ['已读取', 'muted'], ram_verified: ['RAM 已读回验证', 'success'],
        ram_acknowledged: ['RAM 已应答 · 运行值未读回', 'warning'],
        rom_acknowledged: ['ROM 已应答 · 待上电核验', 'warning'], restored: ['备份已恢复', 'success'],
        error: ['参数操作失败', 'error'], unverified: ['待验证', 'warning'],
        write_pending: ['正在写入', 'warning'], write_unverified: ['RAM 写入未验证', 'error'],
        rom_unverified: ['ROM 保存未确认', 'error'], changed_on_device: ['设备参数已变化', 'warning']
      };
      const display = statuses[status] || [status || '未读取', 'muted'];
      $(`${loop}-status`).textContent = display[0];
      $(`${loop}-status`).className = `badge ${display[1]}`;
      const attempt = state?.write_attempts?.[loop];
      const error = attempt?.error;
      const warning = attempt?.warning;
      $(`${loop}-write-error`).textContent = error ? `最近一次写入失败：${error}` : warning ? `写入应答后提示：${warning}` : '';
      $(`${loop}-write-error`).className = error ? 'diagnostics-error' : 'subtle';
      $(`${loop}-write-error`).hidden = !error && !warning;
      const ack = attempt?.ack_ok === true;
      const attemptedReadback = pidFromRaw(attempt?.readback_raw_hex);
      $(`${loop}-write-summary`).textContent = attempt ? `最近目标：${pidText(attempt.requested)}；ACK：${ack ? '已确认' : '未确认'}；${vendorSettings() ? '本次整组设置读回' : '本次参数读回'}：${attemptedReadback ? pidText(attemptedReadback) : '未取得'}` : '';
      $(`${loop}-write-summary`).hidden = !attempt;
      $(`${loop}-write-details`).hidden = !attempt;
      $(`${loop}-write-evidence`).textContent = attempt ? JSON.stringify(attempt, null, 2) : '';
    }
    const current = state?.params?.current;
    $('current-pid').textContent = current ? gains.map(gain => `${gain.toUpperCase()} ${current[gain] ?? '—'}`).join(' / ') : '—';
    $('torque-limit').textContent = state?.params?.torque_limit?.value_raw ?? '—';
    const raws = [...loops, 'current'].map(loop => `${({angle:'角度环',speed:'速度环',current:'电流环'})[loop]}: ${state?.params?.[loop]?.raw_hex || '—'}`);
    raws.push(`扭矩 / 功率限制: ${state?.params?.torque_limit?.raw_hex || '—'}`);
    $('raw-values').textContent = raws.join('\n');
  }

  function refreshControls() {
    const connected = Boolean(state?.connected && serviceHealthy);
    const fresh = (!number(state?.last_feedback_age_s) || state.last_feedback_age_s <= 1) && (!lastReceivedAt || Date.now()-lastReceivedAt <= 2000);
    const stopRequired = Boolean(state?.motion?.stop_required);
    const active = Boolean(state?.motion?.active);
    const successfulHolding = active && state.motion.phase === 'holding' && state.motion.trial_outcome === 'passed' && !stopRequired;
    const tolerance = number(state?.motion?.tolerance_deg) ? state.motion.tolerance_deg : 0.2;
    const stableFeedback = !state?.motion?.hold_warning && number(state?.latest?.angle_deg) && number(state?.motion?.target_deg) && Math.abs(state.latest.angle_deg - state.motion.target_deg) <= tolerance + 1e-9 && number(state?.latest?.speed_deg_s) && Math.abs(state.latest.speed_deg_s) <= 1 && !isFault(state.latest);
    const uncertain = Object.values(state?.pid_status || {}).some(v => ['write_pending','write_unverified','rom_unverified','changed_on_device'].includes(v));
    $('parameter-interlock').hidden = !uncertain;
    $('connect').disabled = connected || commandBusy || !serviceHealthy;
    $('disconnect').disabled = !connected || commandBusy;
    $('read').disabled = !connected || commandBusy || Boolean(state?.motion?.active);
    $('diagnose').disabled = !serviceHealthy || commandBusy || stopRequired || Boolean(state?.motion?.active);
    for (const loop of loops) {
      for (const gain of gains) $(`${loop}-${gain}`).disabled = !connected || !drafts[loop].initialized || commandBusy || Boolean(state?.motion?.active);
      document.querySelector(`.write[data-loop="${loop}"]`).disabled = !connected || !drafts[loop].initialized || commandBusy || stopRequired || Boolean(state?.motion?.active);
      document.querySelector(`.save[data-loop="${loop}"]`).disabled = !connected || commandBusy || stopRequired || uncertain || !['ram_verified', 'ram_acknowledged', 'rom_acknowledged'].includes(state?.pid_status?.[loop]) || Boolean(state?.motion?.active);
      document.querySelector(`.save[data-loop="${loop}"]`).textContent = state?.save_scope === 'all_settings' ? '保存整组 ROM' : '保存 ROM';
      document.querySelector(`.restore[data-loop="${loop}"]`).disabled = !connected || !state?.baseline?.[loop] || commandBusy || stopRequired || Boolean(state?.motion?.active);
    }
    for (const button of document.querySelectorAll('.preset')) button.disabled = !connected || !drafts.angle.initialized || !drafts.speed.initialized || commandBusy || Boolean(state?.motion?.active);
    const enabled = state?.latest?.motor_state === 0;
    $('enable').disabled = !connected || !fresh || commandBusy || stopRequired || uncertain || enabled || Boolean(state?.motion?.active);
    $('enable').textContent = enabled ? '电机已开启' : '使能电机';
    $('move').disabled = !connected || !fresh || !enabled || commandBusy || stopRequired || uncertain || (active && !(successfulHolding && stableFeedback)) || !number(state?.latest?.angle_deg);
    $('move').textContent = successfulHolding ? '前往下一角度' : '▶ 试转并保持观察';
    $('ram-ack-notice').hidden = !vendorSettings() || !Object.keys(state?.ram_ack_values || {}).length || uncertain;
    // Keep stop available during a slow command and after a network failure.
    $('stop').disabled = false;
    $('move-target').disabled = !connected || commandBusy || (active && !successfulHolding);
    $('move-speed').disabled = !connected || commandBusy || (active && !successfulHolding);
    $('use-current').disabled = !connected || !fresh || !number(state?.latest?.angle_deg) || commandBusy || (active && !successfulHolding);
    $('export-json').disabled = !serviceHealthy;
    $('export-csv').disabled = !serviceHealthy;
  }

  function formatTime(value) {
    if (typeof value === 'number' && value < 1e9) return `+${value.toFixed(1)}s`;
    let date;
    if (typeof value === 'number') date = new Date(value > 1e12 ? value : value * 1000);
    else date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value || '—') : date.toLocaleTimeString('zh-CN', {hour12: false});
  }

  function sampleTime(value) {
    if (typeof value === 'number') return value > 1e12 ? value / 1000 : value;
    const converted = Date.parse(value);
    return Number.isNaN(converted) ? Date.now() / 1000 : converted / 1000;
  }

  function isFault(latest) {
    if (!latest) return false;
    const flags = latest.error_flags;
    if (Array.isArray(flags) && flags.length) return true;
    if (typeof flags === 'number' && flags !== 0) return true;
    if (typeof flags === 'string' && !['', '0', '0x00', '0x0000', 'none', 'normal'].includes(flags.toLowerCase())) return true;
    const err = latest.error_state;
    return err !== undefined && err !== null && ![0, false, '', '0', 'none', 'normal', 'ok', '正常', '无故障'].includes(typeof err === 'string' ? err.toLowerCase() : err);
  }

  function renderDiagnostics() {
    const report = state?.diagnostics;
    const signature = report ? JSON.stringify(report) : '';
    if (signature === lastDiagnosticsSignature) return;
    lastDiagnosticsSignature = signature;
    $('diagnostics-empty').hidden = Boolean(report);
    $('diagnostics-meta').hidden = !report;
    $('diagnostics-summary').hidden = !report;
    $('diagnostics-summary').replaceChildren();
    $('diagnostics-steps').replaceChildren();
    if (!report) {
      $('diagnostics-status').textContent = '尚未诊断';
      $('diagnostics-status').className = 'badge muted';
      return;
    }
    const steps = Array.isArray(report.steps) ? report.steps : [];
    const passed = steps.filter(step => step.ok).length;
    $('diagnostics-status').textContent = `${passed} / ${steps.length} 项成功`;
    $('diagnostics-status').className = `badge ${steps.length && passed === steps.length ? 'success' : 'warning'}`;
    const connection = report.connection || {};
    const createdValue = typeof report.created_at === 'number' && report.created_at < 1e12 ? report.created_at * 1000 : report.created_at;
    const created = new Date(createdValue);
    const createdText = Number.isNaN(created.getTime()) ? String(report.created_at || '—') : created.toLocaleString('zh-CN', {hour12: false, timeZone: 'Asia/Shanghai'});
    const mode = report.mode === 'demo' ? '模拟模式' : report.mode === 'real' ? '实机模式' : String(report.mode || '—');
    $('diagnostics-meta').textContent = `${createdText} · ${mode} · ${connection.device || '—'} · 电机 ${connection.motor_id ?? '—'} · ${connection.baudrate ?? '—'} baud`;
    for (const message of Array.isArray(report.summary) ? report.summary : []) {
      const item = document.createElement('li');
      item.textContent = message;
      $('diagnostics-summary').append(item);
    }
    for (const step of steps) {
      const article = document.createElement('article');
      article.className = 'diagnostics-step';
      const heading = document.createElement('div');
      heading.className = 'diagnostics-step-heading';
      const name = document.createElement('strong');
      name.textContent = `${step.name || '读取应答'} · ${step.command || '—'}${step.param_id ? ` / ${step.param_id}` : ''}`;
      const status = document.createElement('span');
      status.className = `badge ${step.ok ? 'success' : 'error'}`;
      status.textContent = step.ok ? '成功' : '失败';
      heading.append(name, status);
      article.append(heading);
      if (step.error) {
        const error = document.createElement('p');
        error.className = 'diagnostics-error';
        error.textContent = step.error;
        article.append(error);
      }
      if (step.decoded !== undefined) {
        const decoded = document.createElement('pre');
        decoded.textContent = JSON.stringify(step.decoded, null, 2);
        article.append(decoded);
      }
      const raw = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = '原始报文与详情';
      const details = document.createElement('pre');
      details.textContent = JSON.stringify({command: step.command, param_id: step.param_id, details: step.details || {}}, null, 2);
      raw.append(summary, details);
      article.append(raw);
      $('diagnostics-steps').append(article);
    }
    // Open each new report once; later polling preserves the user's fold choice.
    $('diagnostics-panel').open = true;
  }

  function renderMotion() {
    const motion = state?.motion || {};
    const active = Boolean(motion.active);
    const holding = active && motion.phase === 'holding';
    const outcome = motion.trial_outcome || (active ? 'pending' : 'idle');
    const passed = outcome === 'passed';
    const warning = holding ? motion.hold_warning : null;
    const tolerance = number(motion.tolerance_deg) ? motion.tolerance_deg : 0.2;
    const verifySeconds = number(motion.verify_hold_s) ? motion.verify_hold_s : 3;
    const verifyTimeout = number(motion.verify_timeout_s) ? motion.verify_timeout_s : 30;
    const stable = number(motion.hold_stable_s) ? motion.hold_stable_s : 0;
    const remaining = number(motion.hold_remaining_s) ? Math.max(0, motion.hold_remaining_s) : null;
    const error = number(motion.last_trial_error_deg) ? motion.last_trial_error_deg : active && number(state?.latest?.angle_deg) && number(motion.target_deg) ? state.latest.angle_deg - motion.target_deg : null;
    const motionResults = {idle:'待机', running:'正在试转', observing:'到位 · 保持验证中', completed:'试验已结束', hold_finished:'保持观察结束 · 已停止', stopped:'已停止', stopped_confirmed:'已确认停稳', stopping:'正在确认停止', stop_unconfirmed:'停止尚未确认', command_failed:'试转指令失败', feedback_failed:'反馈异常后已停止', aborted:'试转已中止', timeout:'试转超时', disconnected:'已断开'};
    let motionText = active ? holding ? passed ? '验证成功 · 仍在保持' : '到位 · 保持验证中' : '正在前往目标' : motionResults[motion.result] || motion.result || '待机';
    let motionClass = active ? passed ? 'success' : 'warning' : 'muted';
    if (warning) { motionText = '保持波动观察中'; motionClass = 'warning'; }
    if (motion.stop_required) { motionText = '停止尚未确认'; motionClass = 'error'; }
    $('motion-status').textContent = motionText;
    $('motion-status').className = `badge ${motionClass}`;
    $('safety-interlock').hidden = !motion.stop_required;
    const results = {idle:['尚未试验','muted'],pending:['验证中','warning'],passed:['验证成功','success'],failed:['验证失败','error'],cancelled:['验证未完成 · 已中止','warning']};
    const result = results[outcome] || [outcome, 'muted'];
    $('trial-outcome').textContent = result[0];
    $('trial-outcome').className = `badge ${result[1]}`;
    $('trial-panel').className = `trial-panel ${result[1]}`;
    if (warning) {
      $('trial-panel').className = 'trial-panel warning';
      $('trial-outcome').className = 'badge warning';
      $('trial-outcome').textContent = passed ? '此前已通过 · 波动观察中' : '验证中 · 波动观察';
    }
    let description = '点击试转后分别记录“到位保持验证”与“停止输出”，停止后保留试验结果。';
    if (active && !holding) description = `前往目标；进入 ±${tolerance}° 且速度 ≤1°/s 后开始保持验证。`;
    else if (holding && !passed) description = `已进入目标附近；须连续 ${verifySeconds} 秒满足误差 ±${tolerance}°、速度 ≤1°/s 才算通过。验证阶段最多 ${verifyTimeout} 秒，当前仍输出保持。`;
    else if (holding && passed) description = `已连续 ${verifySeconds} 秒满足到位条件，现在持续保持同一目标，不按 30 秒停止。可输入下一角度直接继续试转；每个新目标都重新验证 ${verifySeconds} 秒。请留在页面观察。`;
    else if (passed) description = `已通过连续 ${verifySeconds} 秒保持验证。当前保持已结束；停止后的回位不覆盖这次试验结果。`;
    else if (outcome === 'failed') description = '本次到位或保持验证失败；查看停止原因及试验误差，停止后的实际位置请看实时反馈。';
    else if (outcome === 'cancelled') description = '验证完成前已中止，不能据此判定通过。停止后的实际位置请看实时反馈。';
    if (warning) description += ' 当前异常尚在持续性观察中，未发送停止；恢复稳定前暂不切换下一角度。';
    if (motion.stop_required) description = `${passed ? `先前已通过连续 ${verifySeconds} 秒保持验证。` : '试验已中断。'}停止尚未确认，当前输出与保持状态未知；请检查实际运动并尝试停止。`;
    $('trial-description').textContent = description;
    $('trial-start').textContent = `${fmt(motion.start_deg)} °`;
    $('trial-target').textContent = `${fmt(motion.target_deg)} °`;
    $('trial-error-label').textContent = active ? '试验目标误差' : '停止前试验误差';
    $('trial-error').textContent = `${fmt(error)} °`;
    $('trial-peak-error').textContent = `${fmt(motion.hold_peak_error_deg)} °`;
    $('trial-speed-label').textContent = active ? '试验反馈速度' : '停止前试验速度';
    $('trial-speed').textContent = `${fmt(motion.last_trial_speed_deg_s)} °/s`;
    $('trial-peak-speed').textContent = `${fmt(motion.hold_peak_speed_deg_s)} °/s`;
    const delay = number(motion.hold_guard_delay_s) ? motion.hold_guard_delay_s : 0.5;
    const minSamples = motion.hold_guard_min_samples || 3;
    $('hold-guard-policy').textContent = `普通波动连续 ≥${minSamples} 个采样且持续 ≥${delay} 秒才停止；误差达到 ±${motion.hold_hard_error_deg || 1}°或速度达到 ±${motion.hold_hard_speed_deg_s || 10}°/s 立即停止。故障、掉线保护不延迟。`;
    const lastWarning = motion.last_hold_warning;
    $('hold-warning').hidden = !warning && !lastWarning;
    $('hold-warning').textContent = warning
      ? `正在观察：误差 ${fmt(warning.error_deg, 3)}°，速度 ${fmt(warning.speed_deg_s)}°/s；已持续 ${fmt(warning.elapsed_s, 2)} 秒 / ${warning.samples} 个采样。`
      : lastWarning ? `${lastWarning.recovered ? '最近波动已恢复' : '停止前异常记录'}：误差 ${fmt(lastWarning.error_deg, 3)}°，速度 ${fmt(lastWarning.speed_deg_s)}°/s，持续 ${fmt(lastWarning.elapsed_s, 2)} 秒 / ${lastWarning.samples} 个采样。` : '';
    $('hold-progress').hidden = !holding;
    $('hold-stable').textContent = `${fmt(Math.min(stable, verifySeconds), 1)} / ${fmt(verifySeconds, 1)} s`;
    $('hold-stable-progress').max = verifySeconds;
    $('hold-stable-progress').value = Math.min(stable, verifySeconds);
    $('hold-progress-label').textContent = passed ? '保持验证已通过' : '连续稳定验证';
    $('hold-countdown').textContent = passed ? '持续保持 · 可下发下一角度' : `验证剩余 ${fmt(remaining, 1)} 秒 · 可随时手动停止`;
    const reasons = {manual:'手动停止', manual_stop:'手动停止', hold_expired:'保持观察期限到', hold_timeout:'保持观察期限到', hold_limit:'保持观察期限到', timeout:'到位超时', feedback_failed:'反馈异常', heartbeat_timeout:'浏览器心跳中断', heartbeat_lost:'浏览器心跳中断', fault:'电机故障', out_of_bounds:'超出行程限制', hold_unstable:'保持阶段偏离允许范围', disconnected:'已断开连接'};
    $('trial-stop-reason').textContent = motion.stop_reason ? `停止原因：${reasons[motion.stop_reason] || motion.stop_reason}` : '';
    $('trial-stop-reason').hidden = !motion.stop_reason;
    const previous = motion.previous_trial;
    const previousResult = results[previous?.trial_outcome]?.[0] || previous?.trial_outcome || '未完成';
    $('previous-trial').textContent = previous ? `上一目标 ${fmt(previous.target_deg)}°：${previousResult}；最后误差 ${fmt(previous.last_trial_error_deg)}°。完整记录保留在会话 JSON。` : '';
    $('previous-trial').hidden = !previous;
  }

  function renderState(next) {
    const prevConnected = Boolean(state?.connected);
    if (next.session_id && seenSession !== next.session_id) {
      if (seenSession !== null) {
        samples = [];
        sampleSeq = 0;
        lastLogSeq = 0;
        serverLogs = [];
        targetInitialized = false;
        // Keep drafts across a server restart, including values awaiting submission.
        localLog('服务会话已变化。请重新读取电机参数；未提交的表单保留。', 'warning');
      }
      seenSession = next.session_id;
    }
    state = next;
    serviceHealthy = true;
    consecutiveErrors = 0;
    lastReceivedAt = Date.now();
    const connected = Boolean(state.connected);
    if (connected !== prevConnected) {
      if (connected) {
        localLog('电机已连接，当前连接备份由服务端保存。');
      } else {
        localLog('电机连接已断开。参数表单已保留。', 'warning');
      }
    }
    const demo = state.mode === 'demo';
    document.title = `${axisLabel()} PID 调参台`;
    $('axis-heading').textContent = `${axisLabel()} PID 参数调节`;
    $('axis-tag').textContent = `${axisLabel()} · 单电机调试`;
    $('upper-reference').hidden = state.connection?.axis !== 'upper';
    $('mode-badge').textContent = demo ? 'DEMO 模拟模式' : state.mode === 'real' ? '实机模式' : '未知模式';
    $('mode-badge').className = `badge ${demo ? 'demo' : 'real'}`;
    $('mode-notice').className = `notice ${demo ? 'demo-notice' : 'real-notice'}`;
    $('mode-notice-title').textContent = demo ? '模拟演示 · 未连接真实硬件' : '实机调参 · 操作会下发到电机';
    $('mode-notice-text').textContent = demo ? '曲线与参数均为模拟数据。实机模式请通过开发板启动参数选择。' : vendorSettings() ? '整组设置读回与 RAM 写入应答分别记录；应答不等于运行值读回。修改表单后需手动下发。' : '支撑所选关节后再使能。修改表单不会下发，需手动写入 RAM。';
    $('session-label').textContent = `SESSION ${String(state.session_id || '—').slice(0, 8).toUpperCase()}`;
    $('device-value').textContent = state.connection?.device || (demo ? '模拟 RS485' : '—');
    $('motor-id').textContent = state.connection?.motor_id ?? '—';
    $('baudrate').textContent = number(state.connection?.baudrate) ? state.connection.baudrate.toLocaleString('en-US') : (state.connection?.baudrate || '—');
    $('connection-dot').className = connected ? 'on' : '';
    $('connection-label').textContent = connected ? '电机已连接' : '电机未连接';
    $('feedback-status').textContent = connected ? '正在接收' : '未采样';
    renderDiagnostics();
    $('live-dot').className = connected ? 'on pulse' : '';
    $('live-label').textContent = connected ? '界面刷新 · 500 ms' : '等待连接';
    populateParams();
    const latest = state.latest;
    if (number(latest?.t) && latest.t < 1e9) sessionEpoch = Date.now() - latest.t * 1000;
    for (const [id, key, digits] of [['angle-value','angle_deg',2],['speed-value','speed_deg_s',2],['current-value','bus_current_a',2],['voltage-value','voltage_v',1],['temperature-value','temperature_c',0]]) $(id).textContent = fmt(latest?.[key], digits);
    $('target-caption').textContent = `目标 ${fmt(latest?.target_deg)} °`;
    $('speed-last').textContent = `${fmt(latest?.speed_deg_s)} °/s`;
    $('current-last').textContent = `${fmt(latest?.bus_current_a)} A`;
    $('iq-raw').textContent = latest?.iq_or_power_raw ?? '—';
    $('motor-state').textContent = `电机状态 ${latest?.motor_state === 0 ? '已开启 (0x00)' : latest?.motor_state === 16 ? '已关闭 (0x10)' : (latest?.motor_state ?? '—')}`;
    const fault = isFault(latest);
    $('fault-state').className = `fault-state ${!connected || !latest ? 'neutral' : fault ? 'error' : ''}`;
    const faultNames = {low_voltage:'低压保护',high_voltage:'高压保护',driver_overtemperature:'驱动过温',motor_overtemperature:'电机过温',overcurrent:'过流',short_circuit:'短路',stall:'堵转',input_timeout:'输入信号超时'};
    const flagText = Array.isArray(latest?.error_flags) ? latest.error_flags.map(flag => faultNames[flag] || flag).join('、') : String(latest?.error_flags ?? '');
    $('fault-text').textContent = !connected || !latest ? '等待故障状态' : fault ? `故障：${flagText || latest.error_state}` : '未报告驱动故障';
    $('fault-text').title = `error_state: ${latest?.error_state ?? '—'}; flags: ${flagText || '—'}`;
    const limits = state.limits || {};
    $('move-target').min = number(limits.min_deg) ? limits.min_deg : '';
    $('move-target').max = number(limits.max_deg) ? limits.max_deg : '';
    $('move-speed').max = limits.max_test_speed_deg_s || 60;
    if (!targetEdited && !targetInitialized && number(latest?.angle_deg)) {
      const currentAngle = latest.angle_deg;
      let suggested = currentAngle + 2;
      if (number(limits.max_deg) && suggested > limits.max_deg) suggested = currentAngle - 2;
      if (number(limits.min_deg)) suggested = Math.max(suggested, limits.min_deg);
      if (number(limits.max_deg)) suggested = Math.min(suggested, limits.max_deg);
      $('move-target').value = suggested.toFixed(2);
      $('move-speed').value = limits.default_speed_deg_s || 18;
      targetInitialized = true;
    }
    const range = number(limits.min_deg) && number(limits.max_deg) ? ` · 行程 ${fmt(limits.min_deg)}° 至 ${fmt(limits.max_deg)}°` : '';
    $('motion-limits').textContent = `单次位移 ≤ ${limits.max_test_delta_deg || 10}° · 速度 ≤ ${limits.max_test_speed_deg_s || 60}°/s${range}`;
    renderMotion();
    const added = state.readings || [];
    for (const reading of added) {
      if (number(reading.seq) && reading.seq <= sampleSeq) continue;
      const t = sampleTime(reading.t);
      samples.push({...reading, chartTime: t});
      if (number(reading.seq)) sampleSeq = Math.max(sampleSeq, reading.seq);
    }
    if (samples.length) {
      const end = samples[samples.length - 1].chartTime;
      samples = samples.filter(sample => sample.chartTime >= end - 32).slice(-1200);
    }
    const incomingLogs = state.logs || [];
    for (const log of incomingLogs) {
      if (number(log.seq) && log.seq <= lastLogSeq) continue;
      serverLogs.push(log);
      if (number(log.seq)) lastLogSeq = Math.max(lastLogSeq, log.seq);
    }
    serverLogs = serverLogs.slice(-100);
    renderLogs();
    drawCharts();
    const stale = connected && number(state.last_feedback_age_s) && state.last_feedback_age_s > 1;
    document.querySelector('.metrics').classList.toggle('stale', stale || !connected);
    if (stale) {
      $('live-label').textContent = `反馈陈旧 · ${state.last_feedback_age_s.toFixed(1)} s`;
      $('live-dot').className = '';
      $('feedback-status').textContent = '反馈陈旧';
      if (state.motion?.active && !state.motion?.stop_required) {
        $('motion-status').textContent = '反馈陈旧 · 保持状态待确认';
        $('motion-status').className = 'badge warning';
        $('hold-countdown').textContent = '反馈陈旧；显示上次试验状态，尚不能确认仍在保持。';
      }
    }
    if (state.last_error && state.last_error !== lastServerError) showMessage(`驱动通信：${state.last_error}`);
    lastServerError = state.last_error || '';
    refreshControls();
    if ($('token-dialog').open) $('token-dialog').close();
  }

  function renderLogs() {
    const all = [...serverLogs, ...localLogs].sort((a, b) => {
      const logTime = value => typeof value === 'number' ? (value < 1e9 ? sessionEpoch + value * 1000 : value * (value > 1e12 ? 1 : 1000)) : Date.parse(value);
      const at = logTime(a.time);
      const bt = logTime(b.time);
      return (bt || 0) - (at || 0);
    }).slice(0, 60);
    $('log-count').textContent = `${all.length} 条`;
    if (!all.length) return;
    const fragment = document.createDocumentFragment();
    for (const log of all) {
      const li = document.createElement('li');
      const time = document.createElement('time');
      const text = document.createElement('span');
      time.textContent = formatTime(log.time);
      text.textContent = log.message || '';
      li.className = ['error', 'warning'].includes(log.level) ? log.level : '';
      li.append(time, text);
      fragment.append(li);
    }
    $('event-list').replaceChildren(fragment);
  }

  function serviceError(error) {
    serviceHealthy = false;
    consecutiveErrors++;
    $('connection-dot').className = 'error';
    $('connection-label').textContent = '服务连接中断';
    $('feedback-status').textContent = '通信异常';
    $('live-dot').className = '';
    $('live-label').textContent = '数据已暂停';
    if (state?.motion?.active) {
      $('hold-countdown').textContent = '通信中断，保持状态未确认；服务端心跳超时后将尝试停止。';
      if (!state?.motion?.stop_required) {
        $('motion-status').textContent = '通信中断 · 保持状态未知';
        $('motion-status').className = 'badge warning';
      }
    }
    if (consecutiveErrors === 1) {
      showMessage(`${error.message} 页面保留上次读数；停止指令也依赖通信。`);
      localLog(error.message, 'error');
    }
    refreshControls();
  }

  async function poll() {
    if (!token || $('token-dialog').open) return;
    if (pollPromise) return pollPromise;
    pollPromise = (async () => {
      try {
        const next = await request(`/api/state?since=${sampleSeq}`, undefined, {timeout: 6000});
        const wasBroken = !serviceHealthy && consecutiveErrors > 0;
        renderState(next);
        if (wasBroken) {
          showMessage('已恢复与开发板服务的通信。请确认当前电机状态后继续操作。', 'success');
          localLog('开发板服务通信已恢复。');
        }
      } catch (error) { serviceError(error); }
      finally { pollPromise = null; }
    })();
    return pollPromise;
  }

  async function refreshState() {
    if (pollPromise) await pollPromise;
    await poll();
  }

  async function runAction(button, path, data, message, after) {
    if (commandBusy && path !== '/api/stop') return;
    const isStop = path === '/api/stop';
    if (!isStop) commandBusy = true;
    button.classList.add('busy');
    refreshControls();
    try {
      const result = await request(path, data);
      if (after) after(result);
      localLog(message);
      showMessage(message, 'success');
      await refreshState();
      return result;
    } catch (error) {
      showMessage(`${error.message} 操作未确认成功。`);
      localLog(`${error.message}（${path.replace('/api/', '')}）`, 'error');
      if (error.status === 0) serviceError(error);
      // Failed writes may have reached the motor; show the server interlock promptly.
      await refreshState();
    } finally {
      if (!isStop) commandBusy = false;
      button.classList.remove('busy');
      refreshControls();
    }
  }

  for (const loop of loops) {
    for (const gain of gains) $(`${loop}-${gain}`).addEventListener('input', () => markDraft(loop));
    document.querySelector(`.write[data-loop="${loop}"]`).addEventListener('click', async event => {
      try {
        const values = formValues(loop);
        const message = vendorSettings() ? `${nameOf(loop)} RAM 写入已收到成功应答；运行值未独立读回，整组设置读回可能仍为旧值。` : `${nameOf(loop)}已写入 RAM，服务端已读回验证。`;
        await runAction(event.currentTarget, '/api/write', {loop, values}, message, () => {
          // The next state uses the acknowledged target, not the vendor settings cache.
          drafts[loop].dirty = false;
        });
      } catch (error) { showMessage(error.message); }
    });
    document.querySelector(`.save[data-loop="${loop}"]`).addEventListener('click', async event => {
      const button = event.currentTarget;
      const params = state?.params?.[loop];
      if (!params) return;
      const details = gains.map(gain => `${gain.toUpperCase()} = ${params[gain]}`).join('，');
      const dirtyNote = drafts[loop].dirty ? '\n当前表单还有未写入的修改，这些修改不会被保存。' : '';
      if (state.save_scope === 'all_settings') {
        const expected = Object.fromEntries(loops.map(l => {
          const value = state.ram_ack_values?.[l] || state.params?.[l];
          return [l, Object.fromEntries(gains.map(g => [g, value?.[g]]))];
        }));
        if (loops.some(l => gains.some(g => !Number.isInteger(expected[l][g])))) {
          showMessage('两环保存目标不完整，请先读取整组设置。');
          return;
        }
        const both = loops.map(l => `${nameOf(l)}（${state.ram_ack_values?.[l] ? '本会话 RAM 已应答目标' : '整组设置读回'}）：${pidText(expected[l])}`).join('\n');
        if (!await confirmAction('保存整组设置到 ROM？', `将以下两环目标写入整组保存数据：\n${both}\n服务端从最新整组设置快照保留其他字段，叠加本会话已应答目标，再发送整组保存。表单中尚未下发的修改不参与保存；未知参数变化会阻止保存。\nRAM 成功应答不等于运行值已读回，请先人工小幅试转。保存命令成功应答及整组设置核对一致，仍不能证明断电持久；需手动重新上电后核验。重上电前停止并支撑所选关节。`, '确认保存整组 ROM')) return;
        await runAction(button, '/api/save', {loop, confirm: 'SAVE_ALL_ROM', expected}, '整组保存命令已应答且设置读回一致；断电持久性尚待手动重新上电核验。');
        return;
      }
      if (!await confirmAction(`保存${nameOf(loop)}到 ROM？`, `将已读回验证的 RAM 参数发送给 ROM 保存指令：\n${details}${dirtyNote}\n请先完成小幅试转验证。ROM 指令应答不代表已核验断电持久性。\n本手册未给重启指令；按型号需要手动重上电后读回核验持久性。重上电前请停止并支撑所选关节。`, '确认保存 ROM')) return;
      await runAction(button, '/api/save', {loop, confirm: 'SAVE_ROM'}, `${nameOf(loop)} ROM 保存指令已收到应答。`);
    });
    document.querySelector(`.restore[data-loop="${loop}"]`).addEventListener('click', async event => {
      const button = event.currentTarget;
      const baseline = state?.baseline?.[loop];
      if (!baseline) return;
      const details = gains.map(gain => `${gain.toUpperCase()} = ${baseline[gain]}`).join('，');
      const note = vendorSettings() ? '将连接时整组设置快照中的 PID 重新下发 RAM；该快照不是独立读取的原运行 RAM。成功应答仍不代表运行值已读回。' : '将本次连接时保存的参数写回 RAM。';
      if (!await confirmAction(`恢复${nameOf(loop)}连接备份？`, `${note}\n${details}\n请确认所选关节已支撑且当前无运动。恢复不自动保存 ROM。`, '重新下发到 RAM')) return;
      await runAction(button, '/api/restore', {loop}, vendorSettings() ? `${nameOf(loop)}连接快照中的 PID 已重新下发并收到应答；运行值未独立读回。` : `${nameOf(loop)}已恢复本次连接备份到 RAM。`, () => { drafts[loop].dirty = false; });
    });
  }

  for (const button of document.querySelectorAll('.preset')) {
    button.addEventListener('click', () => {
      if (state?.connection?.axis !== 'upper') return;
      $('angle-kp').value = 50;
      $('angle-ki').value = 0;
      $('speed-kp').value = button.dataset.kp;
      $('speed-ki').value = 10;
      for (const loop of loops) markDraft(loop);
      showMessage(`大臂参考值已填入：角度 Kp=50、Ki=0；速度 Kp=${button.dataset.kp}、Ki=10；Kd 保持原值。尚未下发电机。`, 'warning');
    });
  }

  $('connect').addEventListener('click', event => runAction(event.currentTarget, '/api/connect', {}, '电机连接请求已完成。'));
  $('disconnect').addEventListener('click', event => runAction(event.currentTarget, '/api/disconnect', {}, '已断开电机连接。'));
  $('read').addEventListener('click', event => runAction(event.currentTarget, '/api/read', {}, vendorSettings() ? '已更新整组设置读回；保留表单与本会话 RAM 已应答目标，不据此确认运行值。' : loops.some(loop => drafts[loop].dirty) ? '已更新电机读回值；保留了未提交的表单。' : '已从电机重新读取参数。'));
  $('diagnose').addEventListener('click', event => runAction(event.currentTarget, '/api/diagnose', {}, '只读诊断已完成。请查看逐项结果，可下载会话 JSON。'));
  $('enable').addEventListener('click', async event => {
    const button = event.currentTarget;
    if (!await confirmAction(`使能${axisLabel()}电机？`, '使能后电机会输出保持力，可能产生运动。\n确认所选关节已支撑、行程已清空，物理急停可用。', '确认使能')) return;
    await runAction(button, '/api/enable', {confirm:'ENABLE_MOTOR'}, '电机使能指令已确认。');
  });
  $('move-target').addEventListener('input', () => { targetEdited = true; });
  $('use-current').addEventListener('click', () => {
    if (number(state?.latest?.angle_deg)) {
      $('move-target').value = state.latest.angle_deg.toFixed(2);
      targetEdited = true;
    }
  });
  $('move').addEventListener('click', async event => {
    const target = Number($('move-target').value);
    const speed = Number($('move-speed').value);
    const limits = state?.limits || {};
    let error = '';
    if ($('move-target').value.trim() === '' || !Number.isFinite(target)) error = '请输入有效的绝对目标角度。';
    else if ($('move-speed').value.trim() === '' || !Number.isFinite(speed) || speed <= 0 || speed > (limits.max_test_speed_deg_s || 60)) error = `试转速度必须大于 0 且不超过 ${limits.max_test_speed_deg_s || 60}°/s。`;
    else if (!number(state?.latest?.angle_deg)) error = '尚未获得当前位置，无法执行试转。';
    else if (Math.abs(target - state.latest.angle_deg) > (limits.max_test_delta_deg || 10) + 1e-6) error = `目标距当前位置超过 ${limits.max_test_delta_deg || 10}°，请减小单次位移。`;
    else if ((number(limits.min_deg) && target < limits.min_deg) || (number(limits.max_deg) && target > limits.max_deg)) error = '目标角度超出配置的安全行程。';
    if (error) { showMessage(error); return; }
    const button = event.currentTarget;
    const nextAngle = state?.motion?.active && state.motion.phase === 'holding' && state.motion.trial_outcome === 'passed';
    const transition = nextAngle ? '当前保持不中途停止，直接前往新目标；服务端会用最新实测角度重新检查单次位移与限位。\n' : '';
    if (!await confirmAction(nextAngle ? '前往下一角度？' : '试转并持续保持目标？', `${transition}从当前反馈 ${fmt(state?.latest?.angle_deg)}° 前往 ${target}°，速度上限 ${speed}°/s。\n每个目标到位后连续 3 秒满足误差 ±0.2°、速度 ≤1°/s 才显示验证成功，未通过验证的阶段最多 30 秒。通过后持续保持，不按 30 秒停止；可再输入下一角度。\n手动停止、断开连接、保持失败、故障或浏览器心跳中断 3 秒后会尝试停止；停止后保持输出可能解除，机械臂可能回位。请先支撑所选关节、清空行程，并留在页面观察。`, nextAngle ? '确认前往下一角度' : '确认试转并保持')) return;
    await runAction(button, '/api/move', {target_deg: target, speed_deg_s: speed}, `已提交${nextAngle ? '下一角度' : '试转'}：目标 ${target}°，速度 ${speed}°/s；到位后重新验证 3 秒，通过后持续保持。`);
  });
  $('stop').addEventListener('click', event => runAction(event.currentTarget, '/api/stop', {}, '停止指令已确认；保持输出可能解除，注意机械臂回位。停止不会擦除本次试验结果。'));
  $('dismiss-message').addEventListener('click', () => { $('message').hidden = true; });
  $('clear-charts').addEventListener('click', () => { samples = []; drawCharts(); });

  async function download(kind) {
    const button = $(`export-${kind}`);
    button.classList.add('busy');
    button.disabled = true;
    try {
      const blob = await request(`/api/export.${kind}`, undefined, {blob:true});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `arm-pid-${new Date().toISOString().replace(/[:.]/g, '-')}.${kind}`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 3000);
      localLog(`已导出会话 ${kind.toUpperCase()}。`);
    } catch (error) { showMessage(error.message); }
    finally { button.classList.remove('busy'); refreshControls(); }
  }
  $('export-json').addEventListener('click', () => download('json'));
  $('export-csv').addEventListener('click', () => download('csv'));

  function drawChart(id, series, baseRange) {
    const canvas = $(id);
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 3);
    const width = bounds.width;
    const height = bounds.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    const area = {x:42, y:11, w:width - 50, h:height - 34};
    const end = samples.length ? samples[samples.length - 1].chartTime : 30;
    const start = end - 30;
    let min = Infinity;
    let max = -Infinity;
    for (const sample of samples) {
      if (sample.chartTime < start) continue;
      for (const item of series) if (number(sample[item.key])) { min = Math.min(min, sample[item.key]); max = Math.max(max, sample[item.key]); }
    }
    if (!Number.isFinite(min)) [min, max] = baseRange;
    else {
      const span = Math.max(max - min, baseRange[1] - baseRange[0]);
      const center = (max + min) / 2;
      min = center - span * .65;
      max = center + span * .65;
    }
    const span = max - min;
    const precision = span < 2 ? 2 : span < 20 ? 1 : 0;
    ctx.font = '9px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
      const y = area.y + area.h * i / 3;
      ctx.strokeStyle = '#e9eff4';
      ctx.beginPath(); ctx.moveTo(area.x, y); ctx.lineTo(area.x + area.w, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillStyle = '#a5b2bd';
      ctx.fillText((max - span * i / 3).toFixed(precision), area.x - 8, y);
    }
    for (let i = 0; i <= 3; i++) {
      const x = area.x + area.w * i / 3;
      ctx.strokeStyle = '#f1f4f7';
      ctx.beginPath(); ctx.moveTo(x, area.y); ctx.lineTo(x, area.y + area.h); ctx.stroke();
      ctx.textAlign = i === 0 ? 'left' : i === 3 ? 'right' : 'center'; ctx.fillStyle = '#afbbc5';
      ctx.fillText(i === 3 ? '现在' : `${30 - i * 10}s`, x, height - 10);
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(area.x, area.y - 1, area.w + 1, area.h + 2); ctx.clip();
    for (const item of series) {
      const points = samples.filter(sample => sample.chartTime >= start - .5 && number(sample[item.key]));
      const px = sample => area.x + (sample.chartTime - start) / 30 * area.w;
      const py = sample => area.y + (max - sample[item.key]) / span * area.h;
      if (!points.length) continue;
      if (item.fill && points.length > 1) {
        const gradient = ctx.createLinearGradient(0, area.y, 0, area.y + area.h);
        gradient.addColorStop(0, `${item.color}1f`);
        gradient.addColorStop(1, `${item.color}02`);
        ctx.fillStyle = gradient;
        ctx.beginPath(); ctx.moveTo(px(points[0]), area.y + area.h);
        for (const point of points) ctx.lineTo(px(point), py(point));
        ctx.lineTo(px(points[points.length - 1]), area.y + area.h); ctx.closePath(); ctx.fill();
      }
      ctx.beginPath(); ctx.strokeStyle = item.color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
      ctx.setLineDash(item.dashed ? [4, 4] : []);
      let previous = null;
      for (const point of points) {
        if (!previous || point.chartTime - previous.chartTime > 2) ctx.moveTo(px(point), py(point));
        else ctx.lineTo(px(point), py(point));
        previous = point;
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (!item.dashed) {
        const point = points[points.length - 1];
        ctx.beginPath(); ctx.arc(px(point), py(point), 2.6, 0, Math.PI * 2); ctx.fillStyle = item.color; ctx.fill();
      }
    }
    ctx.restore();
  }

  function drawCharts() {
    drawChart('angle-chart', [{key:'target_deg',color:'#91aabf',dashed:true},{key:'angle_deg',color:'#138e87',fill:true}], [-1, 1]);
    drawChart('speed-chart', [{key:'speed_deg_s',color:'#688fbc',fill:true}], [-1, 1]);
    drawChart('current-chart', [{key:'bus_current_a',color:'#b99b62',fill:true}], [0, .2]);
    drawChart('power-chart', [{key:'iq_or_power_raw',color:'#c16a51',fill:true}], [-850, 850]);
    $('angle-empty').hidden = samples.some(sample => number(sample.angle_deg));
    $('speed-empty').hidden = samples.some(sample => number(sample.speed_deg_s));
    $('current-empty').hidden = samples.some(sample => number(sample.bus_current_a));
    $('sample-count').textContent = `${samples.length} 个采样点`;
    $('sample-time').textContent = `最后更新 ${samples.length ? formatTime(samples[samples.length - 1].t) : '—'}`;
  }

  let resizeFrame;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(drawCharts);
  });
  if (window.ResizeObserver) new ResizeObserver(drawCharts).observe($('angle-chart').parentElement);
  window.addEventListener('beforeunload', event => {
    if (loops.some(loop => drafts[loop].dirty) || state?.motion?.active) { event.preventDefault(); event.returnValue = ''; }
  });
  $('token-dialog').addEventListener('cancel', event => { event.preventDefault(); });
  $('token-form').addEventListener('submit', async event => {
    event.preventDefault();
    const value = $('token-input').value.trim();
    if (!value) return;
    rememberToken(value);
    const button = $('token-form').querySelector('button');
    button.disabled = true;
    $('token-error').textContent = '正在验证…';
    try {
      const next = await request(`/api/state?since=${sampleSeq}`);
      renderState(next);
      $('token-input').value = '';
      $('token-dialog').close();
    } catch (error) { $('token-error').textContent = error.message; }
    finally { button.disabled = false; }
  });

  try { token = sessionStorage.getItem('pid_tuner_token') || ''; } catch (_) { /* storage may be disabled */ }
  const hashParams = new URLSearchParams(location.hash.slice(1));
  if (hashParams.get('token')) {
    rememberToken(hashParams.get('token'));
    history.replaceState(null, '', `${location.pathname}${location.search}`);
  }
  refreshControls();
  drawCharts();
  if (!token) askToken();
  else poll();
  setInterval(poll, 500);
  setInterval(() => {
    if (serviceHealthy && state?.connected && lastReceivedAt && Date.now() - lastReceivedAt > 4000) {
      $('live-label').textContent = '反馈延迟';
      $('live-dot').className = '';
    }
    refreshControls();
  }, 1000);
})();
