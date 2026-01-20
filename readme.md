# Distributed Vector Database

A distributed, partition-tolerant vector database system built in Python. This system implements a peer-to-peer cluster architecture using Qdrant as the underlying vector storage engine, supporting vector similarity search with automatic load balancing, replication, and dynamic clustering.

## Features

- **Distributed Architecture**: Multi-node clustering with automatic peer discovery
- **Partition Tolerance**: Hinted handoff and anti-entropy recovery for high availability
- **Phi Accrual Failure Detection**: Probabilistic failure detection based on heartbeat statistics
- **Dynamic Clustering**: K-Means based vector clustering with HNSW index for fast cluster routing
- **Automatic Rebalancing**: Constrained K-Means for balanced cluster distribution
- **Multiple Protocols**: Support for HTTP, gRPC, and QUIC communication
- **Web Interface**: Built-in web client for interactive search

## Project Structure

```
.
├── server.py               # Core distributed server implementation
├── client.py               # Simulation client for testing
├── qdrant_module.py        # Qdrant vector storage abstraction layer
├── cluster_index.py        # HNSW index for fast cluster routing
├── compound_types.py       # Type definitions for vectors and messages
├── generate_peers.py       # Peer configuration generator
│
├── endpoint/               # Server endpoint implementations
│   ├── http_endpoint.py    # HTTP-based peer communication endpoint
│   ├── grpc_endpoint.py    # gRPC-based peer communication endpoint
│   └── quic_endpoint.py    # QUIC-based peer communication endpoint
│
├── communicator/           # Client-side communication
│   ├── http_comm.py        # HTTP communicator for remote calls
│   ├── grpc_comm.py        # gRPC communicator for remote calls
│   └── quic_comm.py        # QUIC communicator for remote calls
│
├── client_endpoint/
│   └── interface.py        # HTTP endpoint for external client interaction
│
├── peer/
│   └── peer.py             # Peer abstraction (local or remote node)
│
├── protos/                 # Protocol Buffer definitions
│   ├── p2p.proto           # P2P message definitions
│   ├── p2p_pb2.py          # Generated protobuf Python code
│   └── p2p_pb2_grpc.py     # Generated gRPC service stubs
│
├── web_client/             # Interactive web interface
│   ├── index.html          # Main HTML page
│   ├── app.js              # Frontend JavaScript logic
│   └── style.css           # Styling
│
├── utils/
│   └── cert_utils.py       # TLS certificate utilities for QUIC
│
├── run.py                  # Docker-based demo launcher
├── startup.py              # Live cluster launcher with web interface
├── benchmark.py            # Performance benchmarking suite
├── tests.py                # Comprehensive test suite
├── requirements.txt        # Python dependencies
└── embeddings.parquet      # Sample embedding dataset
```

## Core Components

### `server.py`
The heart of the system. Implements:
- **Server class**: Main distributed node with vector storage, clustering, and replication
- **VectorStore**: Thread-safe in-memory vector storage with deduplication
- **PhiAccrualFailureDetector**: Adaptive failure detection based on heartbeat intervals
- **HintedHandoff**: Temporary storage for unreachable replicas
- Coordinator election, topology synchronization, and anti-entropy recovery

### `qdrant_module.py`
Abstraction layer over Qdrant vector database. Provides:
- Connection pooling with thread-safe client caching
- Vector CRUD operations (insert, query, delete)
- Support for both in-memory and HTTP-based Qdrant instances

### `cluster_index.py`
HNSW-based index for routing vectors to clusters. Enables O(log n) cluster lookups instead of linear scans.

### `peer/peer.py`
Unified abstraction for local and remote peers. Transparently routes method calls either locally or via network communication.

### `endpoint/` & `communicator/`
Pluggable communication layer supporting:
- **HTTP**: Simple REST-based communication (default)
- **gRPC**: High-performance binary protocol
- **QUIC**: Modern UDP-based transport with TLS

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized Qdrant instances)
- Required Python packages

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Data

The system expects an `embeddings.parquet` file containing vector embeddings. The file should have columns:
- `embedding`: Vector data (list of floats)
- `sentence`: Text payload

## Running the System

### 1. General Demo (`run.py`)

Launches a containerized demo with 8 Qdrant nodes using Docker Compose:

```bash
python run.py
```

**What it does:**
1. Generates `compose.yml` for 8 Qdrant containers
2. Creates storage directories for each node
3. Starts all containers and waits for readiness
4. Runs `client.py` to simulate vector insertion and queries
5. Cleans up all containers and storage on exit

### 2. Live Web Application (`startup.py`)

Launches a live cluster with a web interface for interactive exploration:

```bash
# Launch with 2 nodes (default)
python startup.py

# Launch with custom node count
python startup.py --nodes 4
```

**What it does:**
1. Starts N Qdrant containers with dynamic port allocation
2. Generates `peers.json` configuration file
3. Launches Python server processes for each node
4. Starts a web server for the interactive UI
5. Automatically loads 51,200 vectors from `embeddings.parquet`
6. Opens web client for searching

**Web Interface Features:**
- Vector similarity search with natural language queries
- Real-time results display with similarity scores

### 3. Performance Benchmarking (`benchmark.py`)

Comprehensive benchmarking suite to measure system performance:

```bash
python benchmark.py
```

**Benchmark Scenarios:**

| Scenario | Duration | Threads | Operation Mix | Description |
|----------|----------|---------|---------------|-------------|
| `sanity_check` | 5s | 2 | 50/50 | Quick verification of connectivity |
| `standard` | 30s | 10 | 50/50 | Balanced mixed workload |
| `query_heavy` | 30s | 10 | 10/90 | Read-intensive workload |
| `insert_heavy` | 30s | 10 | 80/20 | Write-intensive workload |
| `stress` | 60s | 20 | 40/60 | High concurrency stress test |

**Metrics Reported:**
- Throughput (operations/second)
- Latency percentiles (P50, P95, P99)
- Error rates
- Per-server performance breakdown

### 4. Test Suite (`tests.py`)

Comprehensive pytest-based test suite covering:

```bash
# Run all tests
pytest tests.py

# Run with verbose output
pytest tests.py -v

# Run specific test class
pytest tests.py::TestServerUnit

# Run with coverage
pytest tests.py --cov=.
```

**Test Categories:**
- **Unit Tests**: Individual component validation
- **Integration Tests**: Multi-node workflow testing
- **Network Partition Tests**: Split-brain and recovery scenarios
- **Failure Detection Tests**: Phi accrual detector behavior
- **Rebalancing Tests**: Cluster split and data migration
- **Anti-Entropy Tests**: Data reconciliation and recovery

## Configuration

### Server Arguments

When running `server.py` directly:

| Argument | Description | Default |
|----------|-------------|---------|
| `--id` | Server ID | 0 |
| `--coordinator` | Flag to designate as coordinator | false |
| `--intra-port` | Port for peer-to-peer communication | 8000 |
| `--inter-port` | Port for client communication | (none) |
| `--qdrant` | Qdrant URL (http://... or :memory:) | :memory: |
| `--endpoint` | Communication protocol (HTTP/GRPC/QUIC) | HTTP |
| `--peers-file` | Path to peers.json configuration | peers.json |

### Peers Configuration

`peers.json` defines the cluster topology. If the system is run with the standard scripts it is automatically created.

```json
[
    {"id": 0, "url": "127.0.0.1", "port": 8000},
    {"id": 1, "url": "127.0.0.1", "port": 8001},
    {"id": 2, "url": "127.0.0.1", "port": 8002}
]
```
