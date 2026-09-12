"use strict";

const state = {
  documents: [],
  currentIndex: -1,
  glossary: {},
  changingPage: false,
};

const elements = {};
const statusLabels = {
  placeholder: "Chưa biên soạn",
  draft: "Đang biên soạn",
  reviewed: "Đã kiểm tra",
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value);
  return node.innerHTML;
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function completedIds() {
  try {
    return new Set(JSON.parse(localStorage.getItem("d2l-completed") || "[]"));
  } catch (_error) {
    return new Set();
  }
}

function saveCompleted(ids) {
  localStorage.setItem("d2l-completed", JSON.stringify([...ids]));
}

function updateProgress() {
  const completed = completedIds();
  const count = state.documents.filter((item) => completed.has(item.id)).length;
  const total = state.documents.length;
  elements.completionText.textContent = `${count} / ${total}`;
  elements.completionBar.style.width = total ? `${(count / total) * 100}%` : "0%";
  document.querySelectorAll(".chapter-link").forEach((link) => {
    link.classList.toggle("done", completed.has(link.dataset.id));
  });
  const current = state.documents[state.currentIndex];
  const isDone = Boolean(current && completed.has(current.id));
  elements.completeButton.setAttribute("aria-pressed", String(isDone));
  elements.completeLabel.textContent = isDone ? "Đã học xong" : "Đánh dấu đã học";
}

function groupLabel(type) {
  return type === "chapter" ? "Các chương" : "Phụ lục";
}

function renderNavigation(filter = "") {
  const query = filter.trim().toLocaleLowerCase("vi");
  const matches = state.documents.filter((item) => {
    const haystack = `${item.number} ${item.title}`.toLocaleLowerCase("vi");
    return haystack.includes(query);
  });
  const completed = completedIds();
  elements.chapterNav.innerHTML = "";

  if (!matches.length) {
    elements.chapterNav.innerHTML = '<p class="chapter-empty">Không tìm thấy chương phù hợp.</p>';
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
      <span class="nav-state" aria-hidden="true"></span>
    `;
    link.addEventListener("click", () => {
      loadDocument(item.id);
      closeSidebar();
    });
    elements.chapterNav.append(link);
  });
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function setLoading() {
  elements.chapterPage.innerHTML = `
    <div class="loading-block wide"></div>
    <div class="loading-block"></div>
    <div class="loading-block short"></div>
  `;
}

function statusLabel(status) {
  return statusLabels[status] || status;
}

function updateDocumentChrome(item) {
  const label = item.type === "chapter" ? "Chương" : "Phụ lục";
  const number = item.type === "chapter" ? String(item.number).padStart(2, "0") : item.number;
  elements.documentKind.textContent = label;
  elements.crumbTitle.textContent = item.title;
  elements.chapterKicker.textContent = `${label.toLocaleUpperCase("vi")} ${number}`;
  elements.pageRange.textContent = `Trang sách ${item.book_pages} · PDF ${item.pdf_pages}`;
  elements.statusBadge.textContent = statusLabel(item.status);
  elements.statusBadge.className = `status-badge ${item.status}`;
  elements.folio.textContent = number;
  elements.positionLabel.textContent = `${state.currentIndex + 1} / ${state.documents.length}`;
  document.title = `${item.title} · D2L Notebook`;

  const previous = state.documents[state.currentIndex - 1];
  const next = state.documents[state.currentIndex + 1];
  elements.previousButton.disabled = !previous;
  elements.footerPrevious.disabled = !previous;
  elements.nextButton.disabled = !next;
  elements.footerNext.disabled = !next;
  elements.previousTitle.textContent = previous?.title || "—";
  elements.nextTitle.textContent = next?.title || "—";
}

function enhanceImages() {
  elements.chapterPage.querySelectorAll("img").forEach((image) => {
    const title = image.getAttribute("title") || image.getAttribute("alt") || "Hình minh họa";
    if (image.parentElement?.tagName === "P") {
      const paragraph = image.parentElement;
      const figure = document.createElement("figure");
      paragraph.replaceWith(figure);
      figure.append(image);
      const caption = document.createElement("figcaption");
      caption.textContent = title;
      figure.append(caption);
    }
    image.addEventListener("click", () => {
      elements.dialogImage.src = image.currentSrc || image.src;
      elements.dialogImage.alt = image.alt;
      elements.dialogCaption.textContent = title;
      elements.imageDialog.showModal();
    });
  });
}

async function typesetMath() {
  if (window.MathJax?.typesetPromise) {
    try {
      await window.MathJax.typesetPromise([elements.chapterPage]);
    } catch (error) {
      console.warn("MathJax could not typeset this chapter", error);
    }
  }
}

async function loadDocument(id, options = {}) {
  const wantedIndex = state.documents.findIndex((item) => item.id === id);
  if (wantedIndex < 0 || state.changingPage) return;
  if (wantedIndex === state.currentIndex && !options.force) return;

  const previousIndex = state.currentIndex;
  const direction = previousIndex < 0 || wantedIndex > previousIndex ? "next" : "previous";
  state.changingPage = true;
  hideTooltip();

  try {
    if (previousIndex >= 0) {
      elements.paperPage.classList.add(`turn-out-${direction}`);
      await sleep(220);
      elements.paperPage.className = "paper-page";
    }
    setLoading();
    const payload = await fetchJson(`/api/chapter/${encodeURIComponent(id)}`);
    state.currentIndex = wantedIndex;
    elements.chapterPage.innerHTML = payload.html;
    updateDocumentChrome(payload);
    enhanceImages();
    renderNavigation(elements.chapterSearch.value);
    updateProgress();
    window.scrollTo({ top: 0, behavior: "instant" });
    elements.paperPage.classList.add(`turn-in-${direction}`);
    await typesetMath();
    window.setTimeout(() => {
      elements.paperPage.className = "paper-page";
    }, 330);

    if (!options.fromHistory) {
      const nextHash = `#${id}`;
      if (options.initial) history.replaceState({ id }, "", nextHash);
      else history.pushState({ id }, "", nextHash);
    }
  } catch (error) {
    showError(`Không thể mở chương: ${error.message}`);
  } finally {
    state.changingPage = false;
  }
}

function adjacentDocument(offset) {
  const target = state.documents[state.currentIndex + offset];
  if (target) loadDocument(target.id);
}

function toggleComplete() {
  const current = state.documents[state.currentIndex];
  if (!current) return;
  const completed = completedIds();
  if (completed.has(current.id)) completed.delete(current.id);
  else completed.add(current.id);
  saveCompleted(completed);
  updateProgress();
}

function showTooltip(target) {
  const key = target.dataset.term;
  const entry = state.glossary[key];
  elements.tooltipTerm.textContent = entry?.term || target.textContent;
  elements.tooltipVi.textContent = entry?.vi ? `· ${entry.vi}` : "· Chưa có chú thích";
  elements.tooltipExplanation.textContent =
    entry?.explanation || "Thuật ngữ này sẽ được giải thích khi chương được biên soạn.";
  elements.tooltipExample.textContent = entry?.example ? `Ví dụ: ${entry.example}` : "";
  elements.tooltipExample.hidden = !entry?.example;
  elements.termTooltip.hidden = false;

  const targetBox = target.getBoundingClientRect();
  const tooltipBox = elements.termTooltip.getBoundingClientRect();
  let left = targetBox.left;
  let top = targetBox.top - tooltipBox.height - 12;
  if (left + tooltipBox.width > window.innerWidth - 10) {
    left = window.innerWidth - tooltipBox.width - 10;
  }
  if (top < 10) top = targetBox.bottom + 12;
  elements.termTooltip.style.left = `${Math.max(10, left)}px`;
  elements.termTooltip.style.top = `${top}px`;
}

function hideTooltip() {
  elements.termTooltip.hidden = true;
}

function openSidebar() {
  elements.sidebar.classList.add("open");
  elements.sidebarScrim.hidden = false;
}

function closeSidebar() {
  elements.sidebar.classList.remove("open");
  elements.sidebarScrim.hidden = true;
}

function showError(message) {
  elements.errorToast.textContent = message;
  elements.errorToast.hidden = false;
  window.setTimeout(() => {
    elements.errorToast.hidden = true;
  }, 5000);
}

function cacheElements() {
  [
    "chapterNav", "chapterSearch", "completionText", "completionBar", "completeButton",
    "completeLabel", "documentKind", "crumbTitle", "chapterKicker", "pageRange",
    "statusBadge", "folio", "positionLabel", "previousButton", "nextButton",
    "footerPrevious", "footerNext", "previousTitle", "nextTitle", "paperPage",
    "chapterPage", "termTooltip", "tooltipTerm", "tooltipVi", "tooltipExplanation",
    "tooltipExample", "sidebar", "sidebarScrim", "menuButton", "sidebarClose",
    "imageDialog", "imageClose", "dialogImage", "dialogCaption", "errorToast",
  ].forEach((id) => {
    elements[id] = byId(id);
  });
}

function registerEvents() {
  elements.chapterSearch.addEventListener("input", (event) => renderNavigation(event.target.value));
  elements.completeButton.addEventListener("click", toggleComplete);
  elements.previousButton.addEventListener("click", () => adjacentDocument(-1));
  elements.footerPrevious.addEventListener("click", () => adjacentDocument(-1));
  elements.nextButton.addEventListener("click", () => adjacentDocument(1));
  elements.footerNext.addEventListener("click", () => adjacentDocument(1));
  elements.menuButton.addEventListener("click", openSidebar);
  elements.sidebarClose.addEventListener("click", closeSidebar);
  elements.sidebarScrim.addEventListener("click", closeSidebar);
  elements.imageClose.addEventListener("click", () => elements.imageDialog.close());
  elements.imageDialog.addEventListener("click", (event) => {
    if (event.target === elements.imageDialog) elements.imageDialog.close();
  });

  elements.chapterPage.addEventListener("pointerover", (event) => {
    const term = event.target.closest(".glossary-term");
    if (term) showTooltip(term);
  });
  elements.chapterPage.addEventListener("pointerout", (event) => {
    if (event.target.closest(".glossary-term")) hideTooltip();
  });
  elements.chapterPage.addEventListener("focusin", (event) => {
    const term = event.target.closest(".glossary-term");
    if (term) showTooltip(term);
  });
  elements.chapterPage.addEventListener("focusout", hideTooltip);
  window.addEventListener("scroll", hideTooltip, { passive: true });

  document.addEventListener("keydown", (event) => {
    const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
    if (event.key === "/" && !typing) {
      event.preventDefault();
      elements.chapterSearch.focus();
      openSidebar();
    } else if (event.key === "ArrowLeft" && !typing) {
      adjacentDocument(-1);
    } else if (event.key === "ArrowRight" && !typing) {
      adjacentDocument(1);
    } else if (event.key === "Escape") {
      closeSidebar();
      hideTooltip();
    }
  });

  window.addEventListener("popstate", () => {
    const id = location.hash.slice(1);
    if (state.documents.some((item) => item.id === id)) {
      loadDocument(id, { fromHistory: true });
    }
  });
}

async function initialize() {
  cacheElements();
  registerEvents();
  try {
    const [notebook, glossary] = await Promise.all([
      fetchJson("/api/notebook"),
      fetchJson("/api/glossary"),
    ]);
    state.documents = notebook.documents;
    state.glossary = glossary.terms || {};
    renderNavigation();
    updateProgress();
    const hashId = location.hash.slice(1);
    const initialId = state.documents.some((item) => item.id === hashId)
      ? hashId
      : state.documents[0]?.id;
    if (initialId) await loadDocument(initialId, { initial: true });
  } catch (error) {
    showError(`Không thể khởi tạo notebook: ${error.message}`);
    elements.chapterPage.innerHTML = `<h1>Không thể mở notebook</h1><p>${escapeHtml(error.message)}</p>`;
  }
}

document.addEventListener("DOMContentLoaded", initialize);
