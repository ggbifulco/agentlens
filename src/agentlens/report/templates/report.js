(function () {
  "use strict";
  // Every value shown below comes from the embedded JSON and is written with
  // textContent / createTextNode, never innerHTML, so trace data cannot inject markup.
  var DATA = { traces: {} };
  try {
    DATA = JSON.parse(document.getElementById("agentlens-data").textContent);
  } catch (err) {
    /* no data: the static report still works */
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  // -- case filters -----------------------------------------------------------------
  var filter = "all";
  var chips = document.querySelectorAll(".chip[data-filter]");
  var search = document.getElementById("case-search");
  var cases = document.querySelectorAll(".case");
  var empty = document.querySelector(".empty-filter");

  function applyFilter() {
    var needle = search ? search.value.trim().toLowerCase() : "";
    var shown = 0;
    cases.forEach(function (item) {
      var ok = (filter === "all" || item.dataset.status === filter) &&
        (!needle || (item.dataset.case || "").toLowerCase().indexOf(needle) !== -1);
      item.hidden = !ok;
      if (ok) shown += 1;
    });
    if (empty) empty.hidden = shown > 0 || cases.length === 0;
  }
  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      filter = chip.dataset.filter;
      chips.forEach(function (c) { c.classList.toggle("active", c === chip); });
      applyFilter();
    });
  });
  if (search) search.addEventListener("input", applyFilter);

  // -- formatting -------------------------------------------------------------------
  function fmtMs(ms) {
    if (ms === null || ms === undefined) return "-";
    if (ms >= 1000) return (ms / 1000).toPrecision(3) + " s";
    if (ms === 0) return "0 ms";
    return ms.toPrecision(3).replace(/\.?0+$/, "") + " ms";
  }
  function fmtCost(cost) {
    return cost === null || cost === undefined ? "-" : "$" + Number(cost.toPrecision(4));
  }

  var TOKEN = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;

  function pretty(value) {
    var pre = el("pre", "json");
    if (typeof value === "string") {
      pre.textContent = value;
      return pre;
    }
    var text = JSON.stringify(value, null, 2);
    if (text === undefined) text = String(value);
    var last = 0;
    var match;
    TOKEN.lastIndex = 0;
    while ((match = TOKEN.exec(text)) !== null) {
      if (match.index > last) pre.appendChild(document.createTextNode(text.slice(last, match.index)));
      var cls = match[1] ? (match[2] ? "j-key" : "j-str") : match[3] ? "j-lit" : "j-num";
      pre.appendChild(el("span", cls, match[1] || match[0]));
      if (match[2]) pre.appendChild(document.createTextNode(match[2]));
      last = TOKEN.lastIndex;
    }
    if (last < text.length) pre.appendChild(document.createTextNode(text.slice(last)));
    return pre;
  }

  function isEmpty(value) {
    if (value === null || value === undefined || value === "") return true;
    if (Array.isArray(value)) return value.length === 0;
    return typeof value === "object" && Object.keys(value).length === 0;
  }

  function section(parent, title, value) {
    if (isEmpty(value)) return;
    parent.appendChild(el("h3", null, title));
    parent.appendChild(pretty(value));
  }

  function keyValues(parent, rows) {
    var dl = el("dl", "kv");
    rows.forEach(function (row) {
      if (row[1] === null || row[1] === undefined || row[1] === "") return;
      dl.appendChild(el("dt", null, row[0]));
      dl.appendChild(el("dd", null, row[1]));
    });
    parent.appendChild(dl);
  }

  // -- trace viewer -----------------------------------------------------------------
  var dialog = document.getElementById("trace-viewer");
  if (!dialog) return;
  // data-inline: the viewer is part of the page (the local UI's trace page), not a modal.
  var inlineId = dialog.dataset.inline || "";
  var tree = dialog.querySelector(".tv-tree");
  var detail = dialog.querySelector(".tv-detail");

  function showRun(trace) {
    detail.replaceChildren();
    var head = el("div", "tv-detail-title");
    head.appendChild(el("span", "badge k-run", "run"));
    head.appendChild(el("span", null, trace.name));
    detail.appendChild(head);
    keyValues(detail, [
      ["case", trace.case_id],
      ["run", trace.case_id === null ? null : "#" + trace.repetition + (trace.side ? " (" + trace.side + ")" : "")],
      ["result", trace.passed ? "passed" : "failed"],
      ["latency", fmtMs(trace.latency_ms)],
      ["cost", fmtCost(trace.cost_usd)],
      ["tokens", String(trace.tokens)],
      ["steps", String(trace.steps)],
      ["started", trace.start],
      ["trace id", trace.id]
    ]);
    if (trace.error) {
      detail.appendChild(el("h3", null, "Error"));
      detail.appendChild(el("pre", "error-box", trace.error));
    }
    if (trace.evals.length) {
      detail.appendChild(el("h3", null, "Evaluations"));
      var list = el("ul", "evals");
      trace.evals.forEach(function (ev) {
        var item = el("li", ev.passed ? "ok" : "ko");
        item.appendChild(el("span", "mark", ev.passed ? "✓" : "✗"));
        item.appendChild(el("b", null, ev.evaluator));
        item.appendChild(document.createTextNode(" "));
        item.appendChild(el("span", "score", ev.score.toFixed(2)));
        if (ev.reason) {
          item.appendChild(document.createTextNode(" "));
          item.appendChild(el("span", "reason", ev.reason));
        }
        list.appendChild(item);
      });
      detail.appendChild(list);
    }
    section(detail, "Input", trace.input);
    section(detail, "Output", trace.output);
    section(detail, "Metadata", trace.metadata);
  }

  function showSpan(span) {
    detail.replaceChildren();
    var head = el("div", "tv-detail-title");
    head.appendChild(el("span", "badge k-" + span.kind, span.kind.replace("_", " ")));
    head.appendChild(el("span", null, span.name));
    detail.appendChild(head);
    var usage = span.usage || {};
    keyValues(detail, [
      ["kind", span.kind],
      ["starts at", "+" + fmtMs(span.offset_ms)],
      ["duration", span.duration_ms === null ? "running (no end time)" : fmtMs(span.duration_ms)],
      ["input tokens", span.usage ? String(usage.input_tokens) : null],
      ["output tokens", span.usage ? String(usage.output_tokens) : null],
      ["cost", span.usage ? fmtCost(usage.cost_usd) : null],
      ["started", span.start],
      ["span id", span.id]
    ]);
    if (span.error) {
      detail.appendChild(el("h3", null, "Error"));
      detail.appendChild(el("pre", "error-box", span.error));
    }
    section(detail, "Input", span.input);
    section(detail, "Output", span.output);
    section(detail, "Attributes", span.attributes);
  }

  function select(row, show) {
    tree.querySelectorAll(".tv-row.selected").forEach(function (r) {
      r.classList.remove("selected");
      r.setAttribute("aria-selected", "false");
    });
    row.classList.add("selected");
    row.setAttribute("aria-selected", "true");
    show();
  }

  function hasError(span) {
    return !!span.error || span.children.some(hasError);
  }

  function node(item, total, depth, show) {
    var wrap = el("div", "tv-node");
    var row = el("div", "tv-row");
    row.tabIndex = 0;
    row.setAttribute("role", "treeitem");
    row.setAttribute("aria-level", String(depth + 1));
    if (item.error) row.classList.add("has-error");
    var label = el("div", "tv-label");
    label.style.paddingLeft = depth * 16 + "px";
    var kids = item.children || [];
    var toggle = el("button", "tv-toggle" + (kids.length ? "" : " leaf"), "▾");
    toggle.type = "button";
    toggle.setAttribute("aria-label", "Collapse");
    label.appendChild(toggle);
    label.appendChild(el("span", "badge k-" + item.kind, item.kind.replace("_", " ")));
    var name = el("span", "tv-name", item.name);
    name.title = item.name;
    label.appendChild(name);
    row.appendChild(label);
    row.appendChild(el("div", "tv-dur", item.duration_ms === null ? "running" : fmtMs(item.duration_ms)));
    var track = el("div", "tv-track");
    var bar = el("div", "tv-bar k-" + item.kind);
    var scale = total > 0 ? total : 1;
    var left = Math.min(100, Math.max(0, (item.offset_ms / scale) * 100));
    var width = item.duration_ms === null ? 100 - left : (item.duration_ms / scale) * 100;
    bar.style.left = left + "%";
    bar.style.width = Math.max(0, Math.min(100 - left, width)) + "%";
    track.appendChild(bar);
    row.appendChild(track);
    wrap.appendChild(row);
    var box = el("div", "tv-kids");
    box.setAttribute("role", "group");
    kids.forEach(function (child) {
      box.appendChild(node(child, total, depth + 1, function () { showSpan(child); }));
    });
    if (kids.length) {
      row.setAttribute("aria-expanded", "true");
      wrap.appendChild(box);
    }
    function setOpen(open) {
      box.classList.toggle("collapsed", !open);
      toggle.textContent = open ? "▾" : "▸";
      toggle.setAttribute("aria-label", open ? "Collapse" : "Expand");
      if (kids.length) row.setAttribute("aria-expanded", String(open));
    }
    wrap.setOpen = setOpen;
    toggle.addEventListener("click", function (ev) {
      ev.stopPropagation();
      setOpen(box.classList.contains("collapsed"));
    });
    row.addEventListener("click", function () { select(row, show); });
    row.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(row, show); }
      else if (ev.key === "ArrowRight" && kids.length) setOpen(true);
      else if (ev.key === "ArrowLeft" && kids.length) setOpen(false);
    });
    return wrap;
  }

  function chip(parent, text, cls) {
    parent.appendChild(el("span", cls || null, text));
  }

  function openTrace(id) {
    var trace = DATA.traces[id];
    if (!trace) return false;
    document.getElementById("tv-title").textContent =
      (trace.case_id === null ? trace.name : trace.case_id + " · run #" + trace.repetition) +
      (trace.side ? " · " + trace.side : "");
    var chips = dialog.querySelector(".tv-chips");
    chips.replaceChildren();
    chip(chips, trace.passed ? "passed" : "failed", trace.passed ? "good" : "bad");
    chip(chips, fmtMs(trace.latency_ms));
    chip(chips, fmtCost(trace.cost_usd));
    chip(chips, trace.tokens + " tokens");
    chip(chips, trace.steps + (trace.steps === 1 ? " step" : " steps"));
    var spans = trace.spans;
    if (trace.error || spans.some(hasError)) chip(chips, "error", "bad");
    var root = {
      kind: "run", name: trace.name, offset_ms: 0, duration_ms: trace.total_ms,
      error: trace.error, children: spans
    };
    tree.replaceChildren(node(root, trace.total_ms, 0, function () { showRun(trace); }));
    select(tree.querySelector(".tv-row"), function () { showRun(trace); });
    if (!dialog.open) {
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
    }
    return true;
  }

  function closeTrace() {
    if (inlineId) return;
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
  }

  dialog.addEventListener("close", function () {
    if (location.hash.indexOf("#trace-") === 0) {
      history.replaceState(null, "", location.pathname + location.search);
    }
  });
  dialog.addEventListener("click", function (ev) {
    var action = ev.target.closest("[data-action]");
    if (!action) {
      if (ev.target === dialog) closeTrace();  // click on the backdrop
      return;
    }
    if (action.dataset.action === "close") closeTrace();
    var open = action.dataset.action === "expand";
    if (action.dataset.action === "expand" || action.dataset.action === "collapse") {
      tree.querySelectorAll(".tv-node").forEach(function (n) { if (n.setOpen) n.setOpen(open); });
      if (!open && tree.firstChild && tree.firstChild.setOpen) tree.firstChild.setOpen(true);
    }
  });
  document.addEventListener("click", function (ev) {
    var link = ev.target.closest("a.open-trace");
    if (!link) return;
    ev.preventDefault();
    if (openTrace(link.dataset.trace)) {
      history.replaceState(null, "", "#trace-" + link.dataset.trace);
    }
  });
  function fromHash() {
    var hash = decodeURIComponent(location.hash || "");
    if (hash.indexOf("#trace-") === 0) openTrace(hash.slice(7));
  }
  if (inlineId) {
    openTrace(inlineId);
  } else {
    window.addEventListener("hashchange", fromHash);
    fromHash();
  }
})();
