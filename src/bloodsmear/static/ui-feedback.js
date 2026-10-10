/* Submission feedback stays local: never render raw error bodies or filenames as HTML. */
(() => {
  'use strict';
  const requests = new WeakMap();
  const messages = {
    UNSUPPORTED_IMAGE_MODE: '暂不支持 16 位灰度图像，请提供经确认的 8 位显微图像。',
    GROUP_SAMPLE_ID_REQUIRED: '同一样本多视野模式需要填写样本 ID，请填写后重新提交。',
    GPU_UNAVAILABLE: 'GPU 当前不可用，请检查显卡驱动和运行环境，或联系维护人员。',
    MODEL_NOT_FOUND: '模型文件缺失，请联系维护人员检查模型部署。',
    MODEL_HASH_MISMATCH: '模型文件校验失败，请联系维护人员检查模型文件。',
    MODEL_VALIDATION_FAILED: '模型配置不符合要求，请联系维护人员检查模型部署。',
    BATCH_TOO_MANY_FILES: '批量图片数量超过限制，请减少图片后重新提交。',
    BATCH_TOTAL_SIZE_EXCEEDED: '整批上传内容超过大小限制，请减少图片或分批提交。',
    JOB_NOT_READY: '批量服务尚未就绪，请稍后重试或联系维护人员。',
  };

  function submissionForm(event) {
    const element = event.detail?.elt || event.target;
    return element?.closest?.('form[data-result-target]') || null;
  }

  function currentRequest(event) {
    const form = submissionForm(event);
    const request = form && requests.get(form);
    return request && request.xhr === event.detail?.xhr ? request : null;
  }

  function show(request, message, error = false, file = null) {
    const panel = document.createElement('div');
    panel.className = `panel feedback-panel ${error ? 'feedback-error' : 'feedback-pending'}`;
    panel.setAttribute('role', error ? 'alert' : 'status');
    const text = document.createElement('p');
    text.textContent = message;
    panel.append(text);
    if (file) {
      const table = document.createElement('table');
      table.className = 'failure-table';
      const head = document.createElement('thead');
      const headings = document.createElement('tr');
      for (const label of ['序号', '文件名', '失败原因']) {
        const cell = document.createElement('th');
        cell.textContent = label;
        headings.append(cell);
      }
      head.append(headings);
      const body = document.createElement('tbody');
      const row = document.createElement('tr');
      for (const value of [file.index, file.filename, file.reason]) {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        row.append(cell);
      }
      body.append(row);
      table.append(head, body);
      panel.append(table);
      const explanation = document.createElement('p');
      explanation.textContent = '其余图片尚未处理，请修正这张图片后重新提交整批。';
      panel.append(explanation);
    } else if (request.files.length) {
      const files = document.createElement('p');
      files.className = 'feedback-files';
      files.textContent = `本次提交：${request.files.join('、')}`;
      panel.append(files);
    }
    // Clearing before sending also detaches any previous batch-status polling element.
    request.target.replaceChildren(panel);
    request.target.scrollIntoView({ block: 'start', behavior: 'auto' });
    request.failed = error;
  }

  function httpFailure(request) {
    const status = request.xhr.status;
    let payload;
    try { payload = JSON.parse(request.xhr.responseText); } catch (_) { /* Bare 500/413 is allowed. */ }
    const suppliedCode = payload?.error?.code;
    const code = typeof suppliedCode === 'string' && /^[A-Z][A-Z0-9_]{0,63}$/.test(suppliedCode)
      ? suppliedCode : `HTTP_${status}`;
    const file = payload?.error?.file;
    if (request.form.dataset.resultTarget === '#batch-result' && file &&
        Number.isInteger(file.index) && file.index >= 1 && file.index <= request.files.length &&
        typeof file.filename === 'string' && typeof file.reason === 'string' &&
        file.filename === request.files[file.index - 1]) {
      show(request, '批量上传未通过校验，整批任务未创建。', true, file);
      return;
    }
    let message = messages[code];
    if (code === 'INVALID_IMAGE') {
      const reason = typeof payload.error.message === 'string' ? payload.error.message : '';
      if (/pixel dimensions/i.test(reason)) {
        message = '图片像素尺寸过大，无法安全读取。请缩小图片或上传普通单视野图片。';
      } else if (/per-file upload limit/i.test(reason)) {
        message = '图片文件大小超过单文件限制，请选择较小的图片。';
      } else if (/exceeds.*25.*MiB/i.test(reason)) {
        message = '图片超过 25 MiB 大小限制，请选择较小的图片。';
      } else if (/unsupported image extension/i.test(reason)) {
        message = '不支持该文件格式，请选择 JPG、JPEG、PNG、TIF 或 TIFF 图片。';
      } else {
        message = '图片无法读取，可能已损坏或内容无效，请重新选择有效的显微图片。';
      }
    }
    if (!message) {
      if (status === 413) message = '上传内容过大，请减少图片大小或分批提交。';
      else if (status === 422) message = '提交信息不完整或格式不正确，请检查图片和填写的信息。';
      else if (status >= 500) message = '服务器处理失败，请稍后重试；如持续出现，请联系维护人员。';
      else message = '提交失败，请检查图片和填写的信息后重新提交。';
    }
    show(request, message, true);
  }

  document.body.addEventListener('htmx:beforeRequest', event => {
    const form = submissionForm(event);
    if (!form) return;
    const previous = requests.get(form);
    if (previous?.pending) {
      event.preventDefault();
      return;
    }
    const target = document.querySelector(form.dataset.resultTarget);
    if (!target) return;
    const files = Array.from(form.querySelectorAll('input[type=file]'))
      .flatMap(input => Array.from(input.files || []).map(file => file.name));
    const request = { form, target, files, xhr: event.detail.xhr, pending: true, failed: false };
    requests.set(form, request);
    form.setAttribute('aria-busy', 'true');
    target.setAttribute('aria-busy', 'true');
    show(request, '处理中，请稍候…');
  });

  document.body.addEventListener('htmx:responseError', event => {
    const request = currentRequest(event);
    if (request) httpFailure(request);
  });

  for (const [name, message] of [
    ['htmx:sendError', '网络连接失败，请检查服务是否启动及网络连接后重试。'],
    ['htmx:timeout', '等待服务器响应超时，请检查服务状态后重试。'],
    ['htmx:sendAbort', '本次请求已取消，未生成新结果，请重新提交。'],
  ]) {
    document.body.addEventListener(name, event => {
      const request = currentRequest(event);
      if (request) show(request, message, true);
    });
  }

  document.body.addEventListener('htmx:afterRequest', event => {
    const request = currentRequest(event);
    if (!request) return;
    request.pending = false;
    request.form.removeAttribute('aria-busy');
    request.target.removeAttribute('aria-busy');
    if (event.detail.failed && !request.failed) httpFailure(request);
  });
})();
