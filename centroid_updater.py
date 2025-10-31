"""
Background updater per centroidi Meta-HNSW.

USO:
- Strategy 'periodic': aggiorna centroidi ogni N secondi
- Funziona in background thread separato
- Nessun impatto su latenza ingestion
"""

import threading
import time
import numpy as np
from typing import Dict
from qdrant_client import QdrantClient
from meta_hnsw import MetaHNSW
from config import COLLECTION_NAME, CENTROID_CALCULATION_METHOD


class CentroidBackgroundUpdater:
    """
    Background updater per centroidi (strategy='periodic').
    
    PROCESSO:
    1. Thread in background ogni N secondi
    2. Fetch sample vettori da ogni nodo Qdrant
    3. Ricalcola centroidi da scratch
    4. Update meta-HNSW
    
    VANTAGGI:
    - Zero overhead su ingestion
    - Centroidi accurati (non approssimati)
    
    SVANTAGGI:
    - Richiede fetch da Qdrant (network I/O)
    - Update meno frequenti (ogni 60s)
    """
    
    def __init__(self, 
                 meta_hnsw: MetaHNSW, 
                 clients: Dict[str, QdrantClient],
                 update_interval: int = 60,
                 sample_size: int = 1000):
        """
        Inizializza background updater.
        
        Args:
            meta_hnsw: Istanza MetaHNSW da aggiornare
            clients: Dictionary {node_name: QdrantClient}
            update_interval: Intervallo update in secondi
            sample_size: Numero vettori da campionare per centroide
        """
        self.meta_hnsw = meta_hnsw
        self.clients = clients
        self.update_interval = update_interval
        self.sample_size = sample_size
        self.running = False
        self.thread = None
        
        print(f"✓ CentroidBackgroundUpdater initialized:")
        print(f"  Update interval: {update_interval}s")
        print(f"  Sample size: {sample_size} vectors/node")
    
    def start(self):
        """Avvia background thread."""
        if self.running:
            print("⚠️  Updater già in esecuzione")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("✓ Background centroid updater started")
    
    def stop(self):
        """Ferma background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✓ Background centroid updater stopped")
    
    def _update_loop(self):
        """Loop principale (esegue in background thread)."""
        while self.running:
            time.sleep(self.update_interval)
            
            if not self.running:
                break
            
            print(f"\n🔄 [{time.strftime('%H:%M:%S')}] Background centroid update...")
            
            for node_name, client in self.clients.items():
                try:
                    # Fetch sample vettori
                    records, _ = client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=self.sample_size,
                        with_vectors=True,
                        with_payload=False
                    )
                    
                    if not records:
                        continue
                    
                    # Estrai vettori
                    vectors = np.array([record.vector for record in records])
                    
                    # Ricalcola centroide
                    self.meta_hnsw.recalculate_centroid_from_scratch(
                        node_name=node_name,
                        vectors=vectors,
                        method=CENTROID_CALCULATION_METHOD
                    )
                    
                except Exception as e:
                    print(f"  ⚠️  Error updating {node_name}: {e}")
            
            print(f"  ✓ Centroid update complete")


def create_background_updater(meta_hnsw: MetaHNSW, 
                              clients: Dict[str, QdrantClient],
                              update_interval: int = 60) -> CentroidBackgroundUpdater:
    """
    Helper per creare e avviare background updater.
    
    Example:
        updater = create_background_updater(meta_hnsw, clients, update_interval=60)
        # ... sistema in esecuzione ...
        updater.stop()
    """
    updater = CentroidBackgroundUpdater(meta_hnsw, clients, update_interval)
    updater.start()
    return updater
