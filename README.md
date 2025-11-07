# progetto-big-data-analytics

# Distributed Qdrant Vector Database

Sistema distribuito di vector database con architettura P2P, clustering intelligente e routing via Meta-HNSW.

## Architettura

**Dual-Stack Communication:**
- **FastAPI (public):** HTTP/JSON per client esterni (porte 8001-800N)
- **gRPC (internal):** Protocol Buffers per comunicazione P2P ad alte prestazioni (porte 9001-900N)

**Vantaggi gRPC:**
- ✅ **10x più veloce** di HTTP+JSON per bulk operations
- ✅ **Streaming bidirezionale** per gossip protocol
- ✅ **Type-safe** con Protocol Buffers
- ✅ **Connessioni persistenti** con multiplexing HTTP/2

## Setup

### 1. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 2. Genera codice gRPC

```bash
python generate_grpc.py
```

Questo genera:
- `generated/vector_service_pb2.py` (messaggi Protocol Buffers)
- `generated/vector_service_pb2_grpc.py` (stubs client/server)

### 3. Avvia il sistema

```bash
python run.py 6  # Avvia 6 nodi
```

Questo avvia:
- **6 Qdrant databases** (Docker, porte 6333-6343)
- **6 FastAPI servers** (porte 8001-8006)
- **6 gRPC servers** (porte 9001-9006)

## Architettura di Rete
