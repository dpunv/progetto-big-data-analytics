import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0";

// State
let embedder = null;
let serverUrl =
  window.SERVER_URL ||
  localStorage.getItem("serverUrl") ||
  "http://localhost:8001";
let topK = parseInt(localStorage.getItem("topK"), 10) || 10;

// DOM Elements
const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const closeSettingsBtn = document.getElementById("close-settings");
const saveSettingsBtn = document.getElementById("save-settings");
const serverUrlInput = document.getElementById("server-url");
const topkInput = document.getElementById("topk-input");
const statusText = document.getElementById("status-text");
const statusDot = document.querySelector(".status-dot");

// Tab Elements
const tabBtns = document.querySelectorAll(".tab-btn");
const viewSections = document.querySelectorAll(".view-section");

// Query Elements
const queryInput = document.getElementById("query-input");
const queryBtn = document.getElementById("query-btn");
const resultsContainer = document.getElementById("results-section");

// Add Elements
const addInput = document.getElementById("add-input");
const addBtn = document.getElementById("add-btn");
const addStatus = document.getElementById("add-status");

// Initialization
async function init() {
  serverUrlInput.value = serverUrl;
  topkInput.value = topK;

  try {
    statusText.textContent = "Loading Model (all-MiniLM-L6-v2)...";
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

    statusText.textContent = "Model Ready";
    statusDot.classList.add("active");
    queryBtn.disabled = false;
    addBtn.disabled = false;
    console.log("Embedder loaded");
  } catch (error) {
    statusText.textContent = "Error loading model";
    console.error("Failed to load embedder:", error);
  }
}

// Tab Switching Logic
tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    // Remove active class from all tabs and views
    tabBtns.forEach((b) => b.classList.remove("active"));
    viewSections.forEach((s) => s.classList.remove("active"));

    // Add active class to clicked tab and target view
    btn.classList.add("active");
    const targetId = btn.getAttribute("data-target");
    document.getElementById(targetId).classList.add("active");
  });
});

// Settings Modal
settingsBtn.addEventListener("click", () => {
  settingsModal.classList.remove("hidden");
  serverUrlInput.value = serverUrl;
  topkInput.value = topK;
});

closeSettingsBtn.addEventListener("click", () => {
  settingsModal.classList.add("hidden");
});

saveSettingsBtn.addEventListener("click", () => {
  let url = serverUrlInput.value.trim();
  if (url.endsWith("/")) {
    url = url.slice(0, -1);
  }
  serverUrl = url;
  localStorage.setItem("serverUrl", serverUrl);

  topK = parseInt(topkInput.value, 10) || 10;
  if (topK < 1) topK = 1;
  if (topK > 100) topK = 100;
  localStorage.setItem("topK", topK);

  settingsModal.classList.add("hidden");
});

// Query Logic
queryBtn.addEventListener("click", performQuery);

queryInput.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") {
    performQuery();
  }
});

async function performQuery() {
  const text = queryInput.value.trim();
  if (!text) return;

  if (!embedder) {
    alert("Model not loaded yet!");
    return;
  }

  queryBtn.disabled = true;
  queryBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
  resultsContainer.innerHTML = "";

  try {
    const startTime = performance.now();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    const embedding = Array.from(output.data);
    const embedTime = (performance.now() - startTime).toFixed(0);

    const response = await fetch(`${serverUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vectors: [embedding], top_k: topK }),
    });

    if (!response.ok) throw new Error(`Server Error: ${response.statusText}`);

    const data = await response.json();

    if (data.results && data.results.length > 0) {
      renderResults(data.results, embedTime);
    } else {
      resultsContainer.innerHTML = '<div class="result-card">No results found.</div>';
    }
  } catch (error) {
    console.error("Query failed:", error);
    resultsContainer.innerHTML = `<div class="result-card" style="border-color: var(--error-color);">Error: ${error.message}</div>`;
  } finally {
    queryBtn.disabled = false;
    queryBtn.innerHTML = '<i class="fa-solid fa-search"></i> Search';
  }
}

function renderResults(results, embedTime) {
  const infoDiv = document.createElement("div");
  infoDiv.className = "status-indicator";
  infoDiv.style.marginBottom = "10px";
  infoDiv.innerHTML = `<small>Embedding generated in ${embedTime}ms. Found ${results.length} matches.</small>`;
  resultsContainer.appendChild(infoDiv);

  results.forEach((res) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const score = res.distance.toFixed(4);

    card.innerHTML = `
            <div class="result-header">
                <span>ID: ${res.id}</span>
                <span class="result-score">Score: ${score}</span>
            </div>
            <div class="result-content">${escapeHtml(res.payload)}</div>
        `;
    resultsContainer.appendChild(card);
  });
}

// Add Logic
addBtn.addEventListener("click", performAdd);

async function performAdd() {
  const text = addInput.value.trim();
  if (!text) return;

  if (!embedder) {
    alert("Model not loaded yet!");
    return;
  }

  addBtn.disabled = true;
  addBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Adding...';
  addStatus.innerHTML = '';
  addStatus.className = '';

  try {
    // 1. Generate Embedding
    const output = await embedder(text, { pooling: "mean", normalize: true });
    const embedding = Array.from(output.data);

    // 2. Send to Server (endpoint expects {"vectors": [[embedding, text], ...]})
    const response = await fetch(`${serverUrl}/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vectors: [[embedding, text]]
      }),
    });

    if (!response.ok) throw new Error(`Server Error: ${response.statusText}`);

    const data = await response.json();

    // Success
    addStatus.innerHTML = '<i class="fa-solid fa-check-circle"></i> Vector added successfully!';
    addStatus.style.color = 'var(--success-color)';
    addInput.value = ''; // Clear input

  } catch (error) {
    console.error("Add failed:", error);
    addStatus.innerHTML = `<i class="fa-solid fa-exclamation-circle"></i> Error: ${error.message}`;
    addStatus.style.color = 'var(--error-color)';
  } finally {
    addBtn.disabled = false;
    addBtn.innerHTML = '<i class="fa-solid fa-plus"></i> Add Vector';
  }
}

function escapeHtml(unsafe) {
  if (typeof unsafe !== "string") return String(unsafe);
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Start
init();
