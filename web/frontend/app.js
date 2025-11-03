const W = () => document.getElementById('graph').clientWidth;
const H = () => document.getElementById('graph').clientHeight;

const svg = d3.select('#graph').append('svg').attr('width', '100%').attr('height', '100%');
const g = svg.append('g');
const zoom = d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => g.attr('transform', e.transform));
svg.call(zoom);

svg.append('defs').append('marker')
    .attr('id', 'arrow').attr('viewBox', '0 -5 10 10').attr('refX', 18).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'var(--link)');

let simulation, nodesSel, linksSel;
const summary = document.getElementById('summary');
const sourcePill = document.getElementById('source');

function degreeMap(edges) {
    const m = new Map();
    edges.forEach(e => { m.set(e.source, (m.get(e.source) || 0) + 1); m.set(e.target, (m.get(e.target) || 0) + 1); });
    return m;
}

function renderGraph(nodes, edges) {
    g.selectAll('*').remove();
    const deg = degreeMap(edges);

    linksSel = g.append('g').attr('stroke-width', 1.5).selectAll('line')
        .data(edges, d => `${d.source}->${d.target}`)
        .join('line')
        .attr('class', 'link');

    const nodeG = g.append('g').selectAll('g')
        .data(nodes, d => d.id)
        .join(enter => {
            const g = enter.append('g').attr('class', 'node');
            g.append('circle')
                .attr('r', d => 10 + (deg.get(d.id) || 1) * 1.5)
                .attr('fill', 'var(--ok)');
            g.append('text').attr('dy', 3).attr('x', 14).text(d => d.label || d.id);
            return g;
        });

    nodesSel = nodeG;

    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(d => d.id).distance(90).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-280))
        .force('center', d3.forceCenter(W() / 2, H() / 2))
        .force('collision', d3.forceCollide().radius(d => 14 + (deg.get(d.id) || 1) * 1.5))
        .on('tick', () => {
            linksSel
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            nodesSel.attr('transform', d => `translate(${d.x},${d.y})`);
        });

    summary.textContent = `${nodes.length} nodes • ${edges.length} links`;
}

function normalizeBase(input) {
    let x = input.trim();
    if (!x) return null;
    if (!/^https?:\/\//i.test(x)) {
        x = 'http://' + x;
    }
    return x.replace(/\/$/, '');
}
function buildUrl(base) { return base + '/get-topology'; }

function parseTopology(json) {
    const topo = json && json.topology ? json.topology : null;
    if (!topo || typeof topo !== 'object') throw new Error('Unexpected format: missing "topology" key');

    const nodes = [];
    const edges = [];
    const seen = new Set();

    for (const [nodeId, info] of Object.entries(topo)) {
        const label = `${nodeId} (${(info.dns || info.ip || '-')}:${info.port ?? '-'})`;
        nodes.push({ id: nodeId, label });

        if (info.peers && typeof info.peers === 'object') {
            for (const [peerId, peerInfo] of Object.entries(info.peers)) {
                const a = nodeId;
                const b = peerId;
                const key = [a, b].sort().join('||');
                if (!seen.has(key)) {
                    seen.add(key);
                    edges.push({ source: a, target: b });
                }
            }
        }
    }
    return { nodes, edges };
}

async function fetchTopology(base) {
    const url = buildUrl(base);
    // Use POST only as requested — the server accepts POST to /get-topology
    const r = await fetch(url, { method: 'POST', mode: 'cors' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    return parseTopology(data);
}

const backdrop = document.getElementById('modal-backdrop');
const hostportInput = document.getElementById('hostport');
const connectBtn = document.getElementById('connect');
const errorBox = document.getElementById('error');
const changeBtn = document.getElementById('change');
const fitBtn = document.getElementById('fit');
const shuffleBtn = document.getElementById('shuffle');

function showModal(prefill) {
    backdrop.style.display = 'flex';
    errorBox.textContent = '';
    hostportInput.value = prefill || '';
    hostportInput.focus();
    hostportInput.select();
}

async function connectFromInput() {
    errorBox.textContent = '';
    const base = normalizeBase(hostportInput.value);
    if (!base) { errorBox.textContent = 'Enter host or host:port'; return; }
    document.getElementById('source').textContent = base + '/get-topology';
    try {
        const { nodes, edges } = await fetchTopology(base);
        renderGraph(nodes, edges);
        backdrop.style.display = 'none';
    } catch (e) {
        errorBox.textContent = 'Connection failed: ' + (e.message || e);
    }
}


connectBtn.addEventListener('click', connectFromInput);
changeBtn.addEventListener('click', () => showModal(hostportInput.value));
hostportInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') connectFromInput(); });
fitBtn.addEventListener('click', () => {
    const bbox = g.node().getBBox();
    const width = W(), height = H();
    const scale = Math.min(4, 0.9 / Math.max(bbox.width / width, bbox.height / height));
    const tx = width / 2 - (bbox.x + bbox.width / 2) * scale;
    const ty = height / 2 - (bbox.y + bbox.height / 2) * scale;
    svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
});
shuffleBtn.addEventListener('click', () => { if (simulation) simulation.alpha(1).restart(); });

showModal('localhost:8001');
