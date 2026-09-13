"use strict";

const state = {
  documents: [], currentIndex: -1, current: null, sourcePages: [], pages: [], pageIndex: 0,
  glossary: {}, changingPage: false, drag: null, documentCache: new Map(),
  layoutCache: new Map(), resizeTimer: 0,
};

const elements = {};
const mobileQuery = window.matchMedia("(max-width: 900px)");
const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
const statusLabels = { placeholder: "Unopened", draft: "In study", reviewed: "Mastered" };

function byId(id) { return document.getElementById(id); }
function visiblePageCount() { return mobileQuery.matches ? 1 : 2; }
function pageStep() { return visiblePageCount(); }
function nextFrame() { return new Promise((resolve) => window.requestAnimationFrame(resolve)); }
function sleep(milliseconds) { return new Promise((resolve) => window.setTimeout(resolve, milliseconds)); }

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value);
  return node.innerHTML;
}

function lastPageStartFor(pages) {
  const count = Math.max(1, pages.length);
  return Math.floor((count - 1) / pageStep()) * pageStep();
}

function lastPageStart() { return lastPageStartFor(state.pages); }

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function completedIds() {
  try { return new Set(JSON.parse(localStorage.getItem("d2l-completed") || "[]")); }
  catch (_error) { return new Set(); }
}

function saveCompleted(ids) { localStorage.setItem("d2l-completed", JSON.stringify([...ids])); }

function updateProgress() {
  const completed = completedIds();
  const count = state.documents.filter((item) => completed.has(item.id)).length;
  const total = state.documents.length;
  elements.completionText.textContent = `${count} / ${total}`;
  elements.completionBar.style.width = total ? `${(count / total) * 100}%` : "0%";
  document.querySelectorAll(".chapter-link").forEach((link) => {
    link.classList.toggle("done", completed.has(link.dataset.id));
  });
  const item = state.documents[state.currentIndex];
  const isDone = Boolean(item && completed.has(item.id));
  elements.completeButton.setAttribute("aria-pressed", String(isDone));
  elements.completeLabel.textContent = isDone ? "Study sealed" : "Seal as studied";
}

function groupLabel(type) { return type === "chapter" ? "Scrolls" : "Appendix scrolls"; }

function renderNavigation(filter = "") {
  const query = filter.trim().toLocaleLowerCase("en");
  const matches = state.documents.filter((item) =>
    `${item.number} ${item.title}`.toLocaleLowerCase("en").includes(query)
  );
  const completed = completedIds();
  elements.chapterNav.innerHTML = "";
  if (!matches.length) {
    elements.chapterNav.innerHTML = '<p class="chapter-empty">No matching scroll.</p>';
    return;
  }
  let previousType = null;
  matches.forEach((item) => {
    if (item.type !== previousType) {
      const heading = document.createElement("p");
      heading.className = "nav-group-title";
      heading.textContent = groupLabel(item.type);
      elements.chapterNav.append(heading);
      previousType = item.type;
    }
    const link = document.createElement("button");
    link.type = "button";
    link.className = "chapter-link";
    link.dataset.id = item.id;
    link.classList.toggle("active", state.documents[state.currentIndex]?.id === item.id);
    link.classList.toggle("done", completed.has(item.id));
    link.setAttribute("aria-current", link.classList.contains("active") ? "page" : "false");
    const number = item.type === "chapter" ? String(item.number).padStart(2, "0") : item.number;
    link.innerHTML = `
      <span class="chapter-number">${escapeHtml(number)}</span>
      <span class="chapter-title">${escapeHtml(item.title)}</span>
      <span class="nav-state" aria-hidden="true"></span>`;
    link.addEventListener("click", () => { loadDocument(item.id); closeSidebar(); });
    elements.chapterNav.append(link);
  });
}

function setLoading() {
  const blocks = '<div class="loading-block wide"></div><div class="loading-block"></div><div class="loading-block short"></div>';
  elements.leftPage.innerHTML = blocks;
  elements.rightPage.innerHTML = blocks;
}

function documentLabel(item) { return item.type === "chapter" ? "Scroll" : "Appendix"; }
function documentNumber(item) { return item.type === "chapter" ? String(item.number).padStart(2, "0") : item.number; }
function blankPageHtml() { return '<div class="blank-note" aria-label="Blank manuscript leaf"><span>End of this scroll</span><i></i><b aria-hidden="true">終</b></div>'; }

function resolveTheme() {
  const saved = localStorage.getItem("d2l-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme, persist = true) {
  document.documentElement.dataset.theme = theme;
  const nextTheme = theme === "dark" ? "light" : "dark";
  const nextThemeLabel = nextTheme === "dark" ? "Night ink" : "Day paper";
  elements.themeIcon.textContent = theme === "dark" ? "☀" : "☾";
  elements.themeLabel.textContent = nextThemeLabel;
  elements.themeButton.setAttribute("aria-label", `Switch to ${nextThemeLabel.toLowerCase()} theme`);
  elements.themeButton.title = `Switch to ${nextThemeLabel.toLowerCase()} theme`;
  if (persist) localStorage.setItem("d2l-theme", theme);
}

function toggleTheme() { applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); }

function loadBookmarks() {
  try {
    const saved = JSON.parse(localStorage.getItem("d2l-bookmarks") || "{}");
    return {
      latest: typeof saved.latest === "string" ? saved.latest : null,
      items: saved.items && typeof saved.items === "object" ? saved.items : {},
    };
  } catch (_error) { return { latest: null, items: {} }; }
}

function saveBookmarks(bookmarks) {
  localStorage.setItem("d2l-bookmarks", JSON.stringify(bookmarks));
}

function normalizedPageText(pageHtml) {
  const holder = document.createElement("div");
  holder.innerHTML = pageHtml || "";
  return (holder.textContent || "").replace(/\s+/g, " ").trim().toLocaleLowerCase("vi");
}

function resolveBookmarkPage(mark, pages) {
  let found = -1;
  if (mark?.anchor) {
    found = pages.findIndex((page) => normalizedPageText(page).includes(mark.anchor));
  }
  if (found < 0) found = Math.min(Number(mark?.pageIndex || 0), Math.max(0, pages.length - 1));
  return Math.floor(found / pageStep()) * pageStep();
}

function updateBookmarkControls() {
  if (!state.current) return;
  const bookmarks = loadBookmarks();
  const currentMark = bookmarks.items[state.current.id];
  const markedPage = currentMark ? resolveBookmarkPage(currentMark, state.pages) : -1;
  const isMarkedHere = markedPage === state.pageIndex;
  elements.bookmarkButton.setAttribute("aria-pressed", String(isMarkedHere));
  elements.bookmarkButton.setAttribute("aria-label", isMarkedHere ? "Untie ribbon from this leaf" : "Tie a ribbon to this leaf");
  elements.bookmarkButton.title = isMarkedHere ? "Untie ribbon from this leaf" : "Tie a ribbon to this leaf";
  elements.bookmarkIcon.textContent = isMarkedHere ? "◆" : "◇";
  elements.bookmarkLabel.textContent = isMarkedHere ? "Ribbon tied" : "Tie ribbon";
  const latest = bookmarks.latest && bookmarks.items[bookmarks.latest];
  const latestIsHere = bookmarks.latest === state.current.id && latest
    && resolveBookmarkPage(latest, state.pages) === state.pageIndex;
  elements.resumeButton.hidden = !latest || latestIsHere;
  elements.notebookSpread.classList.toggle("is-bookmarked", isMarkedHere);
}

function toggleBookmark() {
  if (!state.current) return;
  const bookmarks = loadBookmarks();
  const current = bookmarks.items[state.current.id];
  const currentPage = current ? resolveBookmarkPage(current, state.pages) : -1;
  if (currentPage === state.pageIndex) {
    delete bookmarks.items[state.current.id];
    if (bookmarks.latest === state.current.id) {
      const remaining = Object.entries(bookmarks.items).sort((a, b) => b[1].savedAt - a[1].savedAt);
      bookmarks.latest = remaining[0]?.[0] || null;
    }
  } else {
    bookmarks.items[state.current.id] = {
      pageIndex: state.pageIndex,
      anchor: normalizedPageText(state.pages[state.pageIndex]).slice(0, 160),
      savedAt: Date.now(),
    };
    bookmarks.latest = state.current.id;
  }
  saveBookmarks(bookmarks);
  updateBookmarkControls();
}

async function goToLatestBookmark() {
  const bookmarks = loadBookmarks();
  const documentId = bookmarks.latest;
  const mark = documentId && bookmarks.items[documentId];
  if (!mark) return;
  await loadDocument(documentId, { targetMark: mark });
}

function updateDocumentChrome() {
  const item = state.current;
  if (!item) return;
  const label = documentLabel(item);
  const number = documentNumber(item);
  const leftNumber = state.pageIndex + 1;
  const rightNumber = state.pageIndex + 2;
  elements.documentKind.textContent = label;
  elements.crumbTitle.textContent = item.title;
  elements.leftKicker.textContent = `${label.toUpperCase()} ${number} · LEAF ${String(leftNumber).padStart(2, "0")}`;
  elements.rightKicker.textContent = `${label.toUpperCase()} ${number} · LEAF ${String(rightNumber).padStart(2, "0")}`;
  elements.leftRange.textContent = `Book ${item.book_pages} · PDF ${item.pdf_pages}`;
  elements.rightRange.textContent = rightNumber <= state.pages.length ? "Vietnamese study leaf" : "Blank manuscript leaf";
  elements.spreadHelp.textContent = mobileQuery.matches
    ? "Draw or swipe the paper left for the next leaf; draw right to return."
    : "Draw the right paper leaf left for the next spread, or the left leaf right to return.";
  elements.statusBadge.textContent = statusLabels[item.status] || item.status;
  elements.statusBadge.className = `status-badge ${item.status}`;
  elements.leftFolio.textContent = String(leftNumber).padStart(2, "0");
  elements.rightFolio.textContent = String(rightNumber).padStart(2, "0");
  document.title = `${item.title} · D2L Secret Manual`;
  const visibleEnd = Math.min(state.pageIndex + visiblePageCount(), state.pages.length);
  elements.positionLabel.textContent = visibleEnd === leftNumber
    ? `Leaf ${leftNumber} of ${state.pages.length}`
    : `Leaves ${leftNumber}–${visibleEnd} of ${state.pages.length}`;
  updateNavigationControls();
  updateBookmarkControls();
}

function navigationTarget(offset) {
  const targetPage = state.pageIndex + offset * pageStep();
  if (targetPage >= 0 && targetPage < state.pages.length) return { type: "page", pageIndex: targetPage };
  const targetDocument = state.documents[state.currentIndex + offset];
  return targetDocument ? { type: "document", document: targetDocument } : null;
}

function updateNavigationControls() {
  const previous = navigationTarget(-1);
  const next = navigationTarget(1);
  elements.previousButton.disabled = !previous;
  elements.footerPrevious.disabled = !previous;
  elements.nextButton.disabled = !next;
  elements.footerNext.disabled = !next;
  elements.previousTitle.textContent = previous?.type === "page"
    ? `Leaves ${Math.max(1, state.pageIndex - pageStep() + 1)}–${state.pageIndex}`
    : previous?.document.title || "—";
  elements.nextTitle.textContent = next?.type === "page"
    ? `Leaves ${next.pageIndex + 1}–${Math.min(next.pageIndex + pageStep(), state.pages.length)}`
    : next?.document.title || "—";
}

function decorateImages(container, interactive = true) {
  container.querySelectorAll("img").forEach((image) => {
    const title = image.getAttribute("title") || image.getAttribute("alt") || "Study figure";
    if (image.parentElement?.tagName === "P") {
      const paragraph = image.parentElement;
      const figure = document.createElement("figure");
      paragraph.replaceWith(figure);
      figure.append(image);
      const caption = document.createElement("figcaption");
      caption.textContent = title;
      figure.append(caption);
    }
    image.draggable = false;
    if (!interactive) return;
    image.addEventListener("click", () => {
      elements.dialogImage.src = image.currentSrc || image.src;
      elements.dialogImage.alt = image.alt;
      elements.dialogCaption.textContent = title;
      elements.imageDialog.showModal();
    });
  });
}

async function typesetContainers(containers) {
  if (!window.MathJax?.typesetPromise) return;
  try { await window.MathJax.typesetPromise(containers); }
  catch (error) { console.warn("MathJax could not typeset notebook content", error); }
}

async function waitForImages(container) {
  const pending = [...container.querySelectorAll("img")].filter((image) => !image.complete);
  await Promise.all(pending.map((image) => new Promise((resolve) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", resolve, { once: true });
  })));
}

function probeOverflows(probe) { return probe.scrollHeight > probe.clientHeight + 1; }

function splitListForPage(node, probe) {
  const items = [...node.children];
  if (items.length < 2) return [node];
  const chunks = [];
  let chunk = node.cloneNode(false);
  let ordinal = Number(node.getAttribute("start") || 1);
  items.forEach((item, itemIndex) => {
    chunk.append(item.cloneNode(true));
    probe.replaceChildren(chunk);
    if (probeOverflows(probe) && chunk.children.length > 1) {
      chunk.lastElementChild.remove();
      chunks.push(chunk.cloneNode(true));
      ordinal += chunk.children.length;
      chunk = node.cloneNode(false);
      if (node.tagName === "OL") chunk.setAttribute("start", String(ordinal));
      chunk.append(item.cloneNode(true));
    }
    if (itemIndex === items.length - 1 && chunk.children.length) chunks.push(chunk.cloneNode(true));
  });
  probe.replaceChildren();
  return chunks;
}

function splitPreForPage(node, probe) {
  const code = node.querySelector("code");
  const lines = (code?.textContent || node.textContent || "").split("\n");
  if (lines.length < 3) return [node];
  const chunks = [];
  let currentLines = [];
  lines.forEach((line, lineIndex) => {
    currentLines.push(line);
    const candidate = node.cloneNode(true);
    (candidate.querySelector("code") || candidate).textContent = currentLines.join("\n");
    probe.replaceChildren(candidate);
    if (probeOverflows(probe) && currentLines.length > 1) {
      currentLines.pop();
      const fitted = node.cloneNode(true);
      (fitted.querySelector("code") || fitted).textContent = currentLines.join("\n");
      chunks.push(fitted);
      currentLines = [line];
    }
    if (lineIndex === lines.length - 1 && currentLines.length) {
      const fitted = node.cloneNode(true);
      (fitted.querySelector("code") || fitted).textContent = currentLines.join("\n");
      chunks.push(fitted);
    }
  });
  probe.replaceChildren();
  return chunks;
}

function splitTableForPage(node, probe) {
  const rows = [...node.querySelectorAll("tbody > tr")];
  if (rows.length < 2) return [node];
  const chunks = [];
  let chunk = node.cloneNode(true);
  chunk.querySelectorAll("tbody").forEach((body) => body.replaceChildren());
  rows.forEach((row, rowIndex) => {
    const body = chunk.querySelector("tbody") || chunk.appendChild(document.createElement("tbody"));
    body.append(row.cloneNode(true));
    probe.replaceChildren(chunk);
    if (probeOverflows(probe) && body.children.length > 1) {
      body.lastElementChild.remove();
      chunks.push(chunk.cloneNode(true));
      chunk = node.cloneNode(true);
      chunk.querySelectorAll("tbody").forEach((tableBody) => tableBody.replaceChildren());
      (chunk.querySelector("tbody") || chunk.appendChild(document.createElement("tbody"))).append(row.cloneNode(true));
    }
    if (rowIndex === rows.length - 1) chunks.push(chunk.cloneNode(true));
  });
  probe.replaceChildren();
  return chunks;
}

function splitOversizedNode(node, probe) {
  if (node.matches?.("ul, ol")) return splitListForPage(node, probe);
  if (node.matches?.("pre")) return splitPreForPage(node, probe);
  if (node.matches?.("table")) return splitTableForPage(node, probe);
  node.classList?.add("fit-block");
  return [node];
}

function pushProbePage(probe, pages) {
  const html = probe.innerHTML.trim();
  if (html) pages.push(html);
  probe.replaceChildren();
}

function fillRemainingWithList(node, probe, pages) {
  if (!node.matches?.("ul, ol") || node.children.length < 2) return false;
  const items = [...node.children];
  const fitted = node.cloneNode(false);
  const initialOrdinal = Number(node.getAttribute("start") || 1);
  probe.append(fitted);
  let fittedCount = 0;
  for (const item of items) {
    fitted.append(item.cloneNode(true));
    if (probeOverflows(probe)) {
      fitted.lastElementChild.remove();
      break;
    }
    fittedCount += 1;
  }
  if (!fittedCount) {
    fitted.remove();
    return false;
  }
  pushProbePage(probe, pages);
  if (fittedCount < items.length) {
    const remainder = node.cloneNode(false);
    if (node.tagName === "OL") remainder.setAttribute("start", String(initialOrdinal + fittedCount));
    items.slice(fittedCount).forEach((item) => remainder.append(item.cloneNode(true)));
    addNodeToPages(remainder, probe, pages);
  }
  return true;
}

function addNodeToPages(node, probe, pages) {
  probe.append(node);
  if (!probeOverflows(probe)) return;
  if (probe.childNodes.length > 1) {
    node.remove();
    if (fillRemainingWithList(node, probe, pages)) return;
    const previous = probe.lastElementChild;
    if (previous?.matches("h1, h2, h3, h4")) {
      previous.remove();
      pushProbePage(probe, pages);
      probe.append(previous, node);
      if (!probeOverflows(probe)) return;
      node.remove();
      pushProbePage(probe, pages);
      probe.append(node);
    } else {
      pushProbePage(probe, pages);
      probe.append(node);
    }
  }
  if (!probeOverflows(probe)) return;
  node.remove();
  splitOversizedNode(node, probe).forEach((chunk) => {
    probe.append(chunk);
    if (probeOverflows(probe) && probe.childNodes.length > 1) {
      chunk.remove();
      pushProbePage(probe, pages);
      probe.append(chunk);
    }
    if (probeOverflows(probe)) console.warn("A notebook block is taller than one paper page", chunk);
    pushProbePage(probe, pages);
  });
}

function layoutSignature() {
  const box = elements.leftPage.getBoundingClientRect();
  return `${mobileQuery.matches ? "single" : "spread"}:${Math.round(box.width)}x${Math.round(box.height)}`;
}

async function paginateAuthoredPages(sourcePages) {
  const box = elements.leftPage.getBoundingClientRect();
  if (!box.width || !box.height) return sourcePages;
  const probe = document.createElement("div");
  probe.className = "chapter-page pagination-probe";
  probe.style.width = `${box.width}px`;
  probe.style.height = `${box.height}px`;
  document.body.append(probe);
  const pages = [];
  try {
    for (const sourceHtml of sourcePages) {
      probe.innerHTML = sourceHtml;
      decorateImages(probe, false);
      await typesetContainers([probe]);
      await waitForImages(probe);
      const nodes = [...probe.childNodes].map((node) => node.cloneNode(true));
      probe.replaceChildren();
      nodes.forEach((node) => addNodeToPages(node, probe, pages));
      pushProbePage(probe, pages);
    }
  } finally { probe.remove(); }
  return pages.length ? pages : [blankPageHtml()];
}

function sourcePagesFor(payload) {
  return Array.isArray(payload.pages) && payload.pages.length ? payload.pages : [payload.html];
}

async function layoutPayload(payload) {
  const key = `${payload.id}:${layoutSignature()}`;
  if (!state.layoutCache.has(key)) state.layoutCache.set(key, paginateAuthoredPages(sourcePagesFor(payload)));
  try { return await state.layoutCache.get(key); }
  catch (error) { state.layoutCache.delete(key); throw error; }
}

async function fetchDocument(id) {
  if (!state.documentCache.has(id)) {
    state.documentCache.set(id, fetchJson(`/api/chapter/${encodeURIComponent(id)}`));
  }
  try { return await state.documentCache.get(id); }
  catch (error) { state.documentCache.delete(id); throw error; }
}

async function renderSpread() {
  hideTooltip();
  elements.leftPage.innerHTML = state.pages[state.pageIndex] || blankPageHtml();
  elements.rightPage.innerHTML = state.pages[state.pageIndex + 1] || blankPageHtml();
  decorateImages(elements.leftPage);
  decorateImages(elements.rightPage);
  updateDocumentChrome();
  window.scrollTo({ top: 0, behavior: "instant" });
}

async function commitTarget(targetState) {
  state.currentIndex = targetState.documentIndex;
  state.current = targetState.payload;
  state.sourcePages = sourcePagesFor(targetState.payload);
  state.pages = targetState.pages;
  state.pageIndex = targetState.pageIndex;
  await renderSpread();
  renderNavigation(elements.chapterSearch.value);
  updateProgress();
  prefetchNeighbors();
}

function currentTargetState(pageIndex = state.pageIndex) {
  return { documentIndex: state.currentIndex, payload: state.current, pages: state.pages, pageIndex };
}

async function materializeTarget(target, direction) {
  if (target.type === "page") return currentTargetState(target.pageIndex);
  const payload = await fetchDocument(target.document.id);
  state.documentCache.set(`${target.document.id}:resolved`, payload);
  const pages = await layoutPayload(payload);
  state.layoutCache.set(`${target.document.id}:${layoutSignature()}:resolved`, pages);
  return {
    documentIndex: state.documents.findIndex((item) => item.id === target.document.id),
    payload, pages, pageIndex: direction < 0 ? lastPageStartFor(pages) : 0,
  };
}

function cachedTarget(target, direction) {
  if (target.type === "page") return currentTargetState(target.pageIndex);
  const cacheKey = `${target.document.id}:${layoutSignature()}`;
  const payload = state.documentCache.get(`${target.document.id}:resolved`);
  const pages = state.layoutCache.get(`${cacheKey}:resolved`);
  if (!payload || !pages) return null;
  return {
    documentIndex: state.documents.findIndex((item) => item.id === target.document.id),
    payload, pages, pageIndex: direction < 0 ? lastPageStartFor(pages) : 0,
  };
}

async function prefetchNeighbors() {
  for (const index of [state.currentIndex - 1, state.currentIndex + 1]) {
    const item = state.documents[index];
    if (!item) continue;
    try {
      const payload = await fetchDocument(item.id);
      state.documentCache.set(`${item.id}:resolved`, payload);
      const pages = await layoutPayload(payload);
      state.layoutCache.set(`${item.id}:${layoutSignature()}:resolved`, pages);
    } catch (_error) { /* Optional prefetch; navigation reports real errors. */ }
  }
}

function sheetHtml(item, pages, pageIndex, side) {
  const label = documentLabel(item);
  const number = documentNumber(item);
  const noteNumber = pageIndex + 1;
  const content = pages[pageIndex] || blankPageHtml();
  const range = side === "left" ? `Book ${item.book_pages} · PDF ${item.pdf_pages}`
    : noteNumber <= pages.length ? "Vietnamese study leaf" : "Blank manuscript leaf";
  const badge = side === "left"
    ? `<span class="status-badge ${escapeHtml(item.status)}">${escapeHtml(statusLabels[item.status] || item.status)}</span>`
    : '<span class="page-side-label">墨記</span>';
  const footer = side === "left" ? "D2L · Vietnamese study manuscript" : "Draw or swipe the paper edge";
  return `<header class="page-header"><div><p class="chapter-kicker">${escapeHtml(label.toUpperCase())} ${escapeHtml(number)} · LEAF ${String(noteNumber).padStart(2, "0")}</p><p class="page-range">${escapeHtml(range)}</p></div>${badge}</header><div class="chapter-page">${content}</div><footer class="page-footer"><span>${footer}</span><span>${String(noteNumber).padStart(2, "0")}</span></footer>`;
}

function makeOverlayInert(container) {
  container.querySelectorAll("[id]").forEach((node) => node.removeAttribute("id"));
  container.querySelectorAll("a, button, input, summary, [tabindex]").forEach((node) => node.setAttribute("tabindex", "-1"));
  decorateImages(container, false);
}

function resetFlip() {
  elements.notebookSpread.classList.remove("is-dragging", "is-settling");
  elements.notebookSpread.style.setProperty("--flip-progress", "0");
  elements.flipLeaf.className = "flip-leaf";
  elements.turnUnderlayLeft.classList.remove("active");
  elements.turnUnderlayRight.classList.remove("active");
  elements.flipFront.replaceChildren();
  elements.flipBack.replaceChildren();
  elements.turnUnderlayLeft.replaceChildren();
  elements.turnUnderlayRight.replaceChildren();
}

function prepareFlip(direction, targetState, progress = 0) {
  resetFlip();
  const isMobile = mobileQuery.matches;
  const currentSide = direction > 0 ? "right" : "left";
  const targetSide = direction > 0 ? "left" : "right";
  const currentPage = isMobile ? state.pageIndex : state.pageIndex + (direction > 0 ? 1 : 0);
  const targetPage = isMobile ? targetState.pageIndex : targetState.pageIndex + (direction < 0 ? 1 : 0);
  const underlay = direction > 0 ? elements.turnUnderlayRight : elements.turnUnderlayLeft;
  const underPage = isMobile ? targetState.pageIndex : targetState.pageIndex + (direction > 0 ? 1 : 0);
  const underSide = direction > 0 ? "right" : "left";
  elements.flipFront.innerHTML = sheetHtml(state.current, state.pages, currentPage, currentSide);
  elements.flipBack.innerHTML = sheetHtml(targetState.payload, targetState.pages, targetPage, targetSide);
  underlay.innerHTML = sheetHtml(targetState.payload, targetState.pages, underPage, underSide);
  [elements.flipFront, elements.flipBack, underlay].forEach(makeOverlayInert);
  underlay.classList.add("active");
  elements.flipLeaf.classList.add("active", direction > 0 ? "forward" : "backward");
  elements.notebookSpread.style.setProperty("--flip-progress", progress.toFixed(3));
}

async function settlePreparedFlip(direction, targetState, progress = 0, updateHistory = true) {
  state.changingPage = true;
  hideTooltip();
  elements.notebookSpread.classList.remove("is-dragging");
  elements.notebookSpread.classList.add("is-settling");
  await nextFrame();
  elements.notebookSpread.style.setProperty("--flip-progress", "1");
  await sleep(reducedMotionQuery.matches ? 1 : Math.max(90, Math.round(560 * (1 - progress))));
  const changedDocument = targetState.documentIndex !== state.currentIndex;
  await commitTarget(targetState);
  resetFlip();
  state.changingPage = false;
  if (changedDocument && updateHistory) {
    history.pushState({ id: targetState.payload.id }, "", `#${targetState.payload.id}`);
  }
}

async function cancelPreparedFlip(progress) {
  elements.notebookSpread.classList.remove("is-dragging");
  elements.notebookSpread.classList.add("is-settling");
  await nextFrame();
  elements.notebookSpread.style.setProperty("--flip-progress", "0");
  await sleep(reducedMotionQuery.matches ? 1 : Math.max(100, Math.round(260 * progress)));
  resetFlip();
}

async function loadDocument(id, options = {}) {
  const wantedIndex = state.documents.findIndex((item) => item.id === id);
  if (wantedIndex < 0 || state.changingPage) return;
  if (wantedIndex === state.currentIndex && !options.force) {
    const targetPage = options.targetMark
      ? resolveBookmarkPage(options.targetMark, state.pages)
      : options.openLast ? lastPageStart() : 0;
    if (targetPage === state.pageIndex) { updateBookmarkControls(); return; }
    const direction = targetPage > state.pageIndex ? 1 : -1;
    const targetState = currentTargetState(targetPage);
    prepareFlip(direction, targetState);
    await settlePreparedFlip(direction, targetState, 0, false);
    return;
  }
  const direction = state.currentIndex < 0 || wantedIndex > state.currentIndex ? 1 : -1;
  state.changingPage = true;
  hideTooltip();
  try {
    if (state.currentIndex < 0) setLoading();
    const payload = await fetchDocument(id);
    state.documentCache.set(`${id}:resolved`, payload);
    const pages = await layoutPayload(payload);
    state.layoutCache.set(`${id}:${layoutSignature()}:resolved`, pages);
    const targetState = {
      documentIndex: wantedIndex, payload, pages,
      pageIndex: options.targetMark
        ? resolveBookmarkPage(options.targetMark, pages)
        : options.openLast ? lastPageStartFor(pages) : 0,
    };
    if (state.currentIndex < 0) {
      await commitTarget(targetState);
      state.changingPage = false;
    } else {
      state.changingPage = false;
      prepareFlip(direction, targetState);
      await settlePreparedFlip(direction, targetState, 0, false);
    }
    if (!options.fromHistory) {
      const nextHash = `#${id}`;
      if (options.initial) history.replaceState({ id }, "", nextHash);
      else history.pushState({ id }, "", nextHash);
    }
  } catch (error) {
    resetFlip();
    showError(`Could not open this scroll: ${error.message}`);
  } finally { state.changingPage = false; }
}

async function navigate(offset) {
  if (state.changingPage) return;
  const target = navigationTarget(offset);
  if (!target) return;
  state.changingPage = true;
  try {
    const targetState = await materializeTarget(target, offset);
    state.changingPage = false;
    prepareFlip(offset, targetState);
    await settlePreparedFlip(offset, targetState, 0, true);
  } catch (error) {
    resetFlip();
    state.changingPage = false;
    showError(`Could not turn this paper leaf: ${error.message}`);
  }
}

function toggleComplete() {
  const current = state.documents[state.currentIndex];
  if (!current) return;
  const completed = completedIds();
  if (completed.has(current.id)) completed.delete(current.id); else completed.add(current.id);
  saveCompleted(completed);
  updateProgress();
}

function showTooltip(target) {
  const entry = state.glossary[target.dataset.term];
  elements.tooltipTerm.textContent = entry?.term || target.textContent;
  elements.tooltipVi.textContent = entry?.vi ? `· ${entry.vi}` : "· Note pending";
  elements.tooltipExplanation.textContent = entry?.explanation || "This term does not have a note yet.";
  elements.tooltipExample.textContent = entry?.example ? `Ví dụ: ${entry.example}` : "";
  elements.tooltipExample.hidden = !entry?.example;
  elements.termTooltip.hidden = false;
  const targetBox = target.getBoundingClientRect();
  const tooltipBox = elements.termTooltip.getBoundingClientRect();
  let left = targetBox.left;
  let top = targetBox.top - tooltipBox.height - 12;
  if (left + tooltipBox.width > window.innerWidth - 10) left = window.innerWidth - tooltipBox.width - 10;
  if (top < 10) top = targetBox.bottom + 12;
  elements.termTooltip.style.left = `${Math.max(10, left)}px`;
  elements.termTooltip.style.top = `${top}px`;
}

function hideTooltip() { elements.termTooltip.hidden = true; }
function openSidebar() { elements.sidebar.classList.add("open"); elements.sidebarScrim.hidden = false; }
function closeSidebar() { elements.sidebar.classList.remove("open"); elements.sidebarScrim.hidden = true; }
function showError(message) {
  elements.errorToast.textContent = message;
  elements.errorToast.hidden = false;
  window.setTimeout(() => { elements.errorToast.hidden = true; }, 5000);
}

function beginDrag(event) {
  if (state.changingPage || event.button !== 0) return;
  if (event.target.closest("a, button, input, summary, .glossary-term, img, pre")) return;
  state.drag = { pointerId: event.pointerId, startX: event.clientX, direction: 0, progress: 0, targetState: null };
  elements.notebookSpread.setPointerCapture(event.pointerId);
  elements.notebookSpread.classList.add("is-dragging");
}

function moveDrag(event) {
  if (!state.drag || state.drag.pointerId !== event.pointerId) return;
  const delta = event.clientX - state.drag.startX;
  if (Math.abs(delta) < 3) return;
  const direction = delta < 0 ? 1 : -1;
  const target = navigationTarget(direction);
  if (!target) return;
  let targetState = state.drag.targetState;
  if (direction !== state.drag.direction) {
    targetState = cachedTarget(target, direction);
    if (!targetState) { materializeTarget(target, direction).catch(() => {}); return; }
    prepareFlip(direction, targetState, 0);
    elements.notebookSpread.classList.add("is-dragging");
  }
  const pageWidth = Math.max(260, elements.notebookSpread.clientWidth / visiblePageCount());
  const progress = Math.min(1, Math.abs(delta) / pageWidth);
  state.drag.direction = direction;
  state.drag.progress = progress;
  state.drag.targetState = targetState;
  elements.notebookSpread.style.setProperty("--flip-progress", progress.toFixed(3));
  if (progress > .03) event.preventDefault();
}

function finishDrag(event) {
  if (!state.drag || state.drag.pointerId !== event.pointerId) return;
  const { direction, progress, targetState } = state.drag;
  state.drag = null;
  if (direction && targetState && progress >= .18) {
    settlePreparedFlip(direction, targetState, progress, true).catch((error) => {
      resetFlip(); state.changingPage = false; showError(`Could not turn this paper leaf: ${error.message}`);
    });
  } else if (targetState) cancelPreparedFlip(progress); else resetFlip();
}

async function relayoutCurrentDocument() {
  if (!state.current || state.changingPage || state.drag) return;
  state.changingPage = true;
  const oldCount = Math.max(1, state.pages.length);
  const oldProgress = state.pageIndex / oldCount;
  try {
    const pages = await layoutPayload(state.current);
    state.pages = pages;
    state.pageIndex = Math.min(lastPageStartFor(pages), Math.floor((oldProgress * pages.length) / pageStep()) * pageStep());
    await renderSpread();
    prefetchNeighbors();
  } catch (error) { showError(`Could not reflow these pages: ${error.message}`); }
  finally { state.changingPage = false; }
}

function scheduleRelayout() {
  window.clearTimeout(state.resizeTimer);
  state.resizeTimer = window.setTimeout(relayoutCurrentDocument, 180);
}

function cacheElements() {
  ["chapterNav", "chapterSearch", "completionText", "completionBar", "completeButton", "completeLabel",
    "documentKind", "crumbTitle", "leftKicker", "rightKicker", "leftRange", "rightRange", "statusBadge",
    "leftFolio", "rightFolio", "positionLabel", "previousButton", "nextButton", "footerPrevious", "footerNext",
    "previousTitle", "nextTitle", "notebookSpread", "leftPage", "rightPage", "termTooltip", "tooltipTerm",
    "tooltipVi", "tooltipExplanation", "tooltipExample", "sidebar", "sidebarScrim", "menuButton", "sidebarClose",
    "imageDialog", "imageClose", "dialogImage", "dialogCaption", "errorToast", "spreadHelp", "themeButton",
    "themeIcon", "themeLabel", "bookmarkButton", "bookmarkIcon", "bookmarkLabel", "resumeButton", "resumeLabel",
    "turnUnderlayLeft", "turnUnderlayRight", "flipLeaf", "flipFront", "flipBack",
  ].forEach((id) => { elements[id] = byId(id); });
}

function registerTermEvents(container) {
  container.addEventListener("pointerover", (event) => { const term = event.target.closest(".glossary-term"); if (term) showTooltip(term); });
  container.addEventListener("pointerout", (event) => { if (event.target.closest(".glossary-term")) hideTooltip(); });
  container.addEventListener("focusin", (event) => { const term = event.target.closest(".glossary-term"); if (term) showTooltip(term); });
  container.addEventListener("focusout", hideTooltip);
}

function registerEvents() {
  elements.chapterSearch.addEventListener("input", (event) => renderNavigation(event.target.value));
  elements.themeButton.addEventListener("click", toggleTheme);
  elements.bookmarkButton.addEventListener("click", toggleBookmark);
  elements.resumeButton.addEventListener("click", goToLatestBookmark);
  elements.completeButton.addEventListener("click", toggleComplete);
  elements.previousButton.addEventListener("click", () => navigate(-1));
  elements.footerPrevious.addEventListener("click", () => navigate(-1));
  elements.nextButton.addEventListener("click", () => navigate(1));
  elements.footerNext.addEventListener("click", () => navigate(1));
  elements.menuButton.addEventListener("click", openSidebar);
  elements.sidebarClose.addEventListener("click", closeSidebar);
  elements.sidebarScrim.addEventListener("click", closeSidebar);
  elements.imageClose.addEventListener("click", () => elements.imageDialog.close());
  elements.imageDialog.addEventListener("click", (event) => { if (event.target === elements.imageDialog) elements.imageDialog.close(); });
  registerTermEvents(elements.leftPage);
  registerTermEvents(elements.rightPage);
  window.addEventListener("scroll", hideTooltip, { passive: true });
  elements.notebookSpread.addEventListener("pointerdown", beginDrag);
  elements.notebookSpread.addEventListener("pointermove", moveDrag);
  elements.notebookSpread.addEventListener("pointerup", finishDrag);
  elements.notebookSpread.addEventListener("pointercancel", finishDrag);
  document.addEventListener("keydown", (event) => {
    const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
    if (event.key === "/" && !typing) { event.preventDefault(); elements.chapterSearch.focus(); openSidebar(); }
    else if (event.key === "ArrowLeft" && !typing) navigate(-1);
    else if (event.key === "ArrowRight" && !typing) navigate(1);
    else if (event.key === "Escape") { closeSidebar(); hideTooltip(); }
  });
  window.addEventListener("popstate", () => {
    const id = location.hash.slice(1);
    if (state.documents.some((item) => item.id === id)) loadDocument(id, { fromHistory: true });
  });
  mobileQuery.addEventListener("change", scheduleRelayout);
  window.addEventListener("resize", scheduleRelayout, { passive: true });
}

async function initialize() {
  cacheElements();
  applyTheme(resolveTheme(), false);
  registerEvents();
  try {
    const [notebook, glossary] = await Promise.all([fetchJson("/api/notebook"), fetchJson("/api/glossary")]);
    state.documents = notebook.documents;
    state.glossary = glossary.terms || {};
    renderNavigation();
    updateProgress();
    const requested = location.hash.slice(1);
    const firstId = state.documents.some((item) => item.id === requested) ? requested : state.documents[0]?.id;
    if (firstId) await loadDocument(firstId, { initial: true });
  } catch (error) { showError(`Could not open the study archive: ${error.message}`); }
}

initialize();
