// MONEY MULING DETECTION SYSTEM - Frontend Engine
let network = null;
let currentNodes = new vis.DataSet([]);
let currentEdges = new vis.DataSet([]);
let activePattern = 'dense_core';
let activeThreshold = 0.35;
let activeAccountId = null;

document.addEventListener('DOMContentLoaded', () => {
  initNetwork();
  loadStats();
  loadTopMules();
  loadGraphData();
  loadAlertsTable();
  setupEventListeners();
});

// 1. Initialize Vis-Network
function initNetwork() {
  const container = document.getElementById('network-canvas');
  const data = {
    nodes: currentNodes,
    edges: currentEdges
  };

  const options = {
    nodes: {
      shape: 'dot',
      font: {
        face: 'Inter',
        color: '#f8fafc',
        size: 11,
        strokeWidth: 2,
        strokeColor: '#070a12'
      },
      shadow: {
        enabled: true,
        color: 'rgba(0,0,0,0.6)',
        size: 8,
        x: 2,
        y: 2
      }
    },
    edges: {
      smooth: {
        type: 'continuous',
        roundness: 0.2
      },
      font: {
        face: 'Inter',
        size: 9,
        color: '#94a3b8',
        background: '#070a12',
        strokeWidth: 0
      },
      arrows: {
        to: { enabled: true, scaleFactor: 0.7 }
      }
    },
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -50,
        centralGravity: 0.012,
        springLength: 95,
        springConstant: 0.08,
        damping: 0.45
      },
      stabilization: {
        iterations: 120,
        updateInterval: 25
      }
    },
    interaction: {
      hover: true,
      tooltipDelay: 150,
      zoomView: true,
      dragView: true
    }
  };

  network = new vis.Network(container, data, options);

  // Network Event Listeners
  network.on('click', (params) => {
    if (params.nodes.length > 0) {
      const selectedNodeId = params.nodes[0];
      inspectNode(selectedNodeId);
    } else if (params.edges.length > 0) {
      const selectedEdgeId = params.edges[0];
      inspectEdge(selectedEdgeId);
    }
  });

  network.on('hoverNode', () => {
    container.style.cursor = 'pointer';
  });
  network.on('blurNode', () => {
    container.style.cursor = 'default';
  });
}

// 2. Fetch and Load Graph Data
async function loadGraphData(pattern = activePattern, threshold = activeThreshold, accountId = activeAccountId) {
  showLoader(true);
  try {
    let url = `/api/graph?threshold=${threshold}&pattern=${pattern}&limit=1000`;
    if (accountId) {
      url += `&account_id=${accountId}`;
    }

    const res = await fetch(url);
    const data = await res.json();

    currentNodes.clear();
    currentEdges.clear();

    currentNodes.add(data.nodes);
    currentEdges.add(data.edges);

    // Update live density pill
    const nodeCountEl = document.getElementById('dense-nodes-count');
    const edgeCountEl = document.getElementById('dense-edges-count');
    if (nodeCountEl) nodeCountEl.textContent = data.total_nodes || data.nodes.length;
    if (edgeCountEl) edgeCountEl.textContent = data.total_edges || data.edges.length;

    showLoader(false);

    // Fit view after small layout delay
    setTimeout(() => {
      if (network) {
        network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
      }
    }, 400);

  } catch (err) {
    console.error('Error loading graph:', err);
    showLoader(false);
  }
}

// 3. Load Overall KPIs
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();

    document.getElementById('val-total-trans').textContent = data.total_transactions.toLocaleString();
    document.getElementById('val-flagged-trans').textContent = data.flagged_transactions.toLocaleString();
    
    // Format volume
    const volMillions = (data.total_flagged_volume / 1000000).toFixed(1);
    document.getElementById('val-flagged-vol').textContent = `$${volMillions}M`;
    document.getElementById('val-mules-count').textContent = `${data.suspicious_accounts_count} Hubs`;
    document.getElementById('val-precision').textContent = `${data.top_precision_pct.toFixed(1)}%`;

    // Dataset Status Pill
    const pill = document.getElementById('dataset-pill');
    const nameEl = document.getElementById('dataset-name-text');
    if (pill && nameEl) {
      nameEl.textContent = data.dataset_name || (data.is_custom ? 'Custom Dataset' : 'IBM HI-Small (Default)');
      if (data.is_custom) {
        pill.classList.add('custom-active');
      } else {
        pill.classList.remove('custom-active');
      }
    }
  } catch (err) {
    console.error('Error loading stats:', err);
  }
}


// 4. Load Top Money Mules List (Leaderboard in Default Sidebar)
async function loadTopMules() {
  try {
    const res = await fetch('/api/mules?limit=6');
    const mules = await res.json();

    const listEl = document.getElementById('mules-list');
    if (!listEl) return;

    listEl.innerHTML = '';
    mules.forEach(m => {
      const card = document.createElement('div');
      card.className = 'mule-card';
      card.onclick = () => focusAccount(m.account_id);

      const volStr = m.flagged_volume >= 1000000 
        ? `$${(m.flagged_volume / 1000000).toFixed(1)}M`
        : `$${(m.flagged_volume / 1000).toFixed(0)}K`;

      card.innerHTML = `
        <div>
          <div class="mule-id">Account #${m.account_id}</div>
          <div class="mule-role">${m.role}</div>
        </div>
        <div class="mule-stats">
          <div class="mule-risk">${(m.max_risk * 100).toFixed(1)}% Risk</div>
          <div style="color: #94a3b8; font-size: 11px;">Vol: ${volStr}</div>
        </div>
      `;
      listEl.appendChild(card);
    });
  } catch (err) {
    console.error('Error loading mules:', err);
  }
}

// 5. Inspect Node (Triggered when user clicks a node in the graph)
async function inspectNode(accountId) {
  try {
    const res = await fetch(`/api/node/${accountId}`);
    if (!res.ok) throw new Error('Account not found');
    const data = await res.json();

    // Update panel header
    document.getElementById('panel-heading').textContent = `Account #${data.account_id}`;
    const badge = document.getElementById('panel-badge');
    badge.textContent = data.is_suspicious ? 'SUSPICIOUS ENTITY' : 'NORMAL ACCOUNT';
    badge.className = `panel-badge ${data.is_suspicious ? 'suspicious' : ''}`;

    const content = document.getElementById('panel-content');
    
    // Transactions HTML
    let transHtml = '';
    if (data.transactions && data.transactions.length > 0) {
      transHtml = data.transactions.slice(0, 10).map(t => {
        const isIllicit = t.risk_percent >= 35.0;
        return `
          <div class="trans-item ${isIllicit ? 'illicit' : ''}" onclick="focusEdge(${t.edge_id})">
            <div class="trans-flow">
              <span>${t.direction === 'OUTGOING' ? '↗ To' : '↙ From'} #${t.counterparty}</span>
              <span class="flow-arrow">&bull;</span>
              <span style="color: #64748b;">${t.format}</span>
            </div>
            <div>
              <span class="trans-amount">$${t.amount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
              <span class="trans-risk ${isIllicit ? 'highlight-red' : ''}">(${t.risk_percent}%)</span>
            </div>
          </div>
        `;
      }).join('');
    } else {
      transHtml = '<div style="color: #64748b; font-size: 12px;">No recorded transactions</div>';
    }

    content.innerHTML = `
      <div class="entity-header-card">
        <div class="entity-title-row">
          <span class="entity-id">Account #${data.account_id}</span>
          <span class="risk-pill ${data.is_suspicious ? '' : 'normal'}">${data.max_risk_percent}% Max Risk</span>
        </div>
        <div class="entity-role-tag">Role: <strong>${data.role}</strong></div>

        <div class="stats-grid-2">
          <div class="mini-stat">
            <div class="mini-stat-label">Total Outflow</div>
            <div class="mini-stat-value" style="color: #f87171;">$${data.out_volume.toLocaleString()}</div>
            <div style="font-size: 10px; color: #64748b;">${data.out_count} transfers (${data.flagged_out} flagged)</div>
          </div>
          <div class="mini-stat">
            <div class="mini-stat-label">Total Inflow</div>
            <div class="mini-stat-value" style="color: #38bdf8;">$${data.in_volume.toLocaleString()}</div>
            <div style="font-size: 10px; color: #64748b;">${data.in_count} transfers (${data.flagged_in} flagged)</div>
          </div>
        </div>
      </div>

      <div style="display: flex; gap: 8px;">
        <button class="btn-primary" style="flex: 1;" onclick="focusAccountNeighborhood(${data.account_id})">🔍 Focus Ego Network</button>
        <button class="btn-subtle" onclick="resetSidebar()">Close</button>
      </div>

      <div>
        <div class="trans-history-header">Recent Transaction Stream (Top 10)</div>
        <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 6px;">
          ${transHtml}
        </div>
      </div>
    `;

    // Highlight the node in the canvas
    if (network) {
      network.selectNodes([accountId]);
    }

  } catch (err) {
    console.error('Error inspecting node:', err);
  }
}

// 6. Inspect Edge (When user clicks a transaction edge)
function inspectEdge(edgeId) {
  const edge = currentEdges.get(edgeId);
  if (!edge) return;

  document.getElementById('panel-heading').textContent = `Transaction #${edge.id}`;
  const badge = document.getElementById('panel-badge');
  badge.textContent = edge.is_suspicious ? 'HIGH-RISK TRANSFER' : 'STANDARD TRANSFER';
  badge.className = `panel-badge ${edge.is_suspicious ? 'suspicious' : ''}`;

  const content = document.getElementById('panel-content');
  content.innerHTML = `
    <div class="entity-header-card">
      <div class="entity-title-row">
        <span class="entity-id">Tx #${edge.id}</span>
        <span class="risk-pill">${edge.risk}% Risk</span>
      </div>
      <div class="entity-role-tag">Payment Channel: <strong>${edge.format}</strong> &bull; Currency: <strong>${edge.currency}</strong></div>

      <div class="stats-grid-2">
        <div class="mini-stat">
          <div class="mini-stat-label">From Account</div>
          <div class="mini-stat-value" style="cursor: pointer; color: #38bdf8;" onclick="inspectNode(${edge.from})">Acc #${edge.from} ↗</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-label">To Account</div>
          <div class="mini-stat-value" style="cursor: pointer; color: #38bdf8;" onclick="inspectNode(${edge.to})">Acc #${edge.to} ↘</div>
        </div>
      </div>

      <div class="mini-stat" style="margin-top: 6px;">
        <div class="mini-stat-label">Transaction Amount</div>
        <div class="mini-stat-value" style="font-size: 18px; color: #f8fafc;">$${edge.amount.toLocaleString(undefined, {minimumFractionDigits: 2})}</div>
      </div>
    </div>

    <div style="display: flex; gap: 8px;">
      <button class="btn-primary" style="flex: 1;" onclick="focusAccount(${edge.from})">Inspect Sender (#${edge.from})</button>
      <button class="btn-subtle" style="flex: 1;" onclick="focusAccount(${edge.to})">Inspect Receiver (#${edge.to})</button>
    </div>
  `;
}

// 7. Load Bottom Alerts Feed Table
async function loadAlertsTable(minRisk = activeThreshold) {
  try {
    const res = await fetch(`/api/alerts?min_risk=${minRisk}&page_size=20`);
    const data = await res.json();

    document.getElementById('alerts-count-badge').textContent = `${data.total.toLocaleString()} Active Alerts`;
    const tbody = document.getElementById('alerts-tbody');
    tbody.innerHTML = '';

    data.alerts.forEach(a => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="table-mono">#${a.edge_id}</td>
        <td class="table-mono" onclick="inspectNode(${a.from_id})" style="color: #38bdf8;">#${a.from_id}</td>
        <td style="color: #64748b;">➔</td>
        <td class="table-mono" onclick="inspectNode(${a.to_id})" style="color: #38bdf8;">#${a.to_id}</td>
        <td class="amount-col">$${a.amount.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
        <td>${a.format}</td>
        <td>${a.currency}</td>
        <td class="risk-col">${a.risk_percent}%</td>
        <td><span class="pill-tag ${a.is_laundering ? '' : 'normal'}">${a.is_laundering ? 'ILLICIT AML' : 'NORMAL'}</span></td>
        <td><button class="btn-inspect" onclick="focusAlertInGraph(${a.from_id}, ${a.to_id}, ${a.edge_id})">Inspect Graph</button></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Error loading alerts table:', err);
  }
}

// 8. Focus an Account on the Graph
function focusAccount(accountId) {
  activeAccountId = accountId;
  inspectNode(accountId);
  loadGraphData(activePattern, activeThreshold, accountId);
}

function focusAccountNeighborhood(accountId) {
  focusAccount(accountId);
}

function focusAlertInGraph(fromId, toId, edgeId) {
  focusAccount(fromId);
}

function resetSidebar() {
  document.getElementById('panel-heading').textContent = 'Entity Intelligence';
  const badge = document.getElementById('panel-badge');
  badge.textContent = 'Overview';
  badge.className = 'panel-badge';
  
  const content = document.getElementById('panel-content');
  content.innerHTML = `
    <div class="default-view">
      <div class="helper-box">
        <span class="helper-icon">💡</span>
        <p>Click any <strong>node</strong> or <strong>transaction edge</strong> in the graph to inspect detailed AML risk metrics and transaction history.</p>
      </div>

      <div class="mule-leaderboard">
        <h3 class="subheading">Top Flagged Money Mule Hubs</h3>
        <p class="subheading-desc">Identified by high-centrality multi-hop flows:</p>
        <div id="mules-list" class="mules-list-container"></div>
      </div>
    </div>
  `;
  loadTopMules();
}

// 9. UI Event Listeners
function setupEventListeners() {
  // Pattern Pills
  document.querySelectorAll('.preset-pill').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.preset-pill').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      activePattern = e.target.dataset.pattern;
      activeAccountId = null;
      loadGraphData(activePattern, activeThreshold, null);
    });
  });

  // Threshold Slider
  const slider = document.getElementById('threshold-slider');
  const thresholdVal = document.getElementById('threshold-val');
  slider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    activeThreshold = val / 100.0;
    thresholdVal.textContent = `${val}%`;
  });
  slider.addEventListener('change', () => {
    loadGraphData(activePattern, activeThreshold, activeAccountId);
    loadAlertsTable(activeThreshold);
  });

  // Search Box
  const searchInput = document.getElementById('account-search-input');
  const searchBtn = document.getElementById('search-btn');

  const doSearch = () => {
    const query = searchInput.value.trim();
    if (query && !isNaN(query)) {
      focusAccount(parseInt(query));
    }
  };

  searchBtn.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doSearch();
  });

  // Fit Screen Button
  document.getElementById('fit-graph-btn').addEventListener('click', () => {
    if (network) network.fit({ animation: { duration: 500 } });
  });

  // Reset Button
  document.getElementById('refresh-btn').addEventListener('click', () => {
    activeAccountId = null;
    searchInput.value = '';
    resetSidebar();
    loadGraphData(activePattern, activeThreshold, null);
  });
}

// 10. Toast Notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  
  let icon = 'ℹ️';
  if (type === 'success') icon = '⚡';
  if (type === 'error') icon = '⚠️';

  toast.innerHTML = `
    <span style="font-size: 16px;">${icon}</span>
    <span style="flex: 1; line-height: 1.4;">${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => {
      if (toast.parentElement) toast.remove();
    }, 400);
  }, 4500);
}

function showLoader(show) {
  const loader = document.getElementById('graph-loader');
  if (!loader) return;
  if (show) {
    loader.classList.remove('hidden');
  } else {
    loader.classList.add('hidden');
  }
}


