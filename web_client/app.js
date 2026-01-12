import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0";

// State
let embedder = null;
// Use server URL from config.js (set by startup.py) if available, else localStorage, else default
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
const queryInput = document.getElementById("query-input");
const queryBtn = document.getElementById("query-btn");
const topkInput = document.getElementById("topk-input");
const resultsContainer = document.getElementById("results-section");
const statusText = document.getElementById("status-text");
const statusDot = document.querySelector(".status-dot");

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
    console.log("Embedder loaded");
  } catch (error) {
    statusText.textContent = "Error loading model";
    console.error("Failed to load embedder:", error);
  }
}

// Event Listeners
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
  // Remove trailing slash if present
  if (url.endsWith("/")) {
    url = url.slice(0, -1);
  }
  serverUrl = url;
  localStorage.setItem("serverUrl", serverUrl);

  // Save topK setting
  topK = parseInt(topkInput.value, 10) || 10;
  if (topK < 1) topK = 1;
  if (topK > 100) topK = 100;
  localStorage.setItem("topK", topK);

  settingsModal.classList.add("hidden");
});

queryBtn.addEventListener("click", performQuery);

// Allow Ctrl+Enter to submit
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

  // UI Loading state
  queryBtn.disabled = true;
  queryBtn.innerHTML =
    '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
  resultsContainer.innerHTML = ""; // Clear previous

  try {
    // 1. Generate Embedding
    const startTime = performance.now();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    const embedding = Array.from(output.data);
    const embedTime = (performance.now() - startTime).toFixed(0);

    // 2. Send to Server
    const response = await fetch(`${serverUrl}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        vectors: [embedding],
        top_k: topK,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server Error: ${response.statusText}`);
    }

    const data = await response.json();

    // 3. Render Results
    if (data.results && data.results.length > 0) {
      renderResults(data.results, embedTime);
    } else {
      resultsContainer.innerHTML =
        '<div class="result-card">No results found.</div>';
    }
  } catch (error) {
    console.error("Query failed:", error);
    resultsContainer.innerHTML = `<div class="result-card" style="border-color: var(--error-color);">Error: ${error.message}</div>`;
  } finally {
    queryBtn.disabled = false;
    queryBtn.innerHTML = '<i class="fa-solid fa-search"></i> Query';
  }
}

function renderResults(results, embedTime) {
  // Info header
  const infoDiv = document.createElement("div");
  infoDiv.className = "status-indicator";
  infoDiv.style.marginBottom = "10px";
  infoDiv.innerHTML = `<small>Embedding generated in ${embedTime}ms. Found ${results.length} matches.</small>`;
  resultsContainer.appendChild(infoDiv);

  results.forEach((res) => {
    const card = document.createElement("div");
    card.className = "result-card";

    // Format distance as percentage or score
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
