import random
import socket
import struct
import threading
import time

# --- CONFIGURAZIONE ---
NODES_COUNT = 4
BASE_UDP_PORT = 7000
HEARTBEAT_INTERVAL = 2.0  # Veloce per il test (in prod: 5.0)
TIMEOUT_LIMIT = 10.0  # Se non ti sento per 6s, sei morto (in prod: 30.0)
JITTER = 1.0  # Jitter +/- 1s

# Costanti Carico
ALPHA = 1.5e-7
BETA = 4e-4


# Colori per il terminale
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def log(node_id, msg, color=Colors.OKBLUE):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"{color}[{timestamp}] [NODE-{node_id}] {msg}{Colors.ENDC}")


class MockNode:
    def __init__(self, node_id, total_nodes):
        self.node_id = node_id
        self.port = BASE_UDP_PORT + node_id
        self.peers_ports = [
            BASE_UDP_PORT + i for i in range(1, total_nodes + 1) if i != node_id
        ]

        # Stato
        self.running = False
        self.sock = None

        # Metriche Simulate
        self.num_vectors = 0
        self.qps = 0
        self.current_load = 0.5

        # Tabella Salute: { peer_id: {'load': float, 'last_seen': time, 'status': str} }
        self.peer_table = {}
        # Inizializza la tabella vuota/offline per i peer conosciuti
        for p_port in self.peers_ports:
            pid = p_port - BASE_UDP_PORT
            self.peer_table[pid] = {"load": 0.0, "last_seen": 0, "status": "unknown"}

    def start(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind alla porta localhost specifica
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.setblocking(False)

        log(self.node_id, f"Started on UDP port {self.port}", Colors.OKGREEN)

        # Thread Ricezione
        threading.Thread(target=self._listener, daemon=True).start()
        # Thread Invio (Heartbeat)
        threading.Thread(target=self._sender, daemon=True).start()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        log(self.node_id, "CRASHED / STOPPED!", Colors.FAIL)

    def _calculate_simulated_load(self):
        """Simula variazioni di traffico e dati"""
        # Random Walk: aggiunge/toglie vettori e QPS
        self.num_vectors += random.randint(0, 5000)
        self.qps = max(0, self.qps + random.randint(-50, 50))
        if self.qps < 10:
            self.qps = 100  # Keep alive traffic

        load = 0.5 + (ALPHA * self.num_vectors) + (BETA * self.qps)
        return load

    def _listener(self):
        while self.running:
            try:
                # Ricezione
                data, addr = self.sock.recvfrom(1024)

                if len(data) >= 8:
                    sender_id, sender_load = struct.unpack("!if", data[:8])

                    old_status = self.peer_table.get(sender_id, {}).get("status")
                    self.peer_table[sender_id] = {
                        "load": sender_load,
                        "last_seen": time.time(),
                        "status": "ONLINE",
                    }

                    if old_status != "ONLINE":
                        log(
                            self.node_id,
                            f"Detected NODE-{sender_id} is back ONLINE! (Load: {sender_load:.2f})",
                            Colors.OKGREEN,
                        )

            except BlockingIOError:
                # Nessun dato disponibile (normale per socket non bloccanti)
                time.sleep(0.1)

            except ConnectionResetError:
                # [FIX WINDOWS] Ignora errore se inviamo a una porta chiusa
                # Questo è l'errore che uccideva il thread!
                pass

            except OSError:
                # Se la socket è stata chiusa volontariamente (self.running=False), usciamo.
                # Altrimenti stampiamo l'errore ma NON usciamo se è un errore di rete strano.
                if not self.running:
                    break
                else:
                    # Logghiamo l'errore ma continuiamo ad ascoltare!
                    # print(f"Socket error on Node {self.node_id}: {e}")
                    pass
            except Exception as e:
                print(f"Generic error on Node {self.node_id}: {e}")

    def _sender(self):
        while self.running:
            try:
                # 1. Calcola Carico
                self.current_load = self._calculate_simulated_load()

                # 2. Pack Payload
                payload = struct.pack("!if", self.node_id, self.current_load)

                # 3. Invia a tutti i peer
                for p_port in self.peers_ports:
                    self.sock.sendto(payload, ("127.0.0.1", p_port))

                # 4. Controllo Timeout (Failure Detector)
                self._check_failures()

                # 5. Sleep con Jitter
                jitter_val = random.uniform(-JITTER / 2, JITTER / 2)
                time.sleep(HEARTBEAT_INTERVAL + jitter_val)

            except OSError:
                break  # Socket chiusa

    def _check_failures(self):
        now = time.time()
        for pid, info in self.peer_table.items():
            if info["status"] == "ONLINE":
                if (now - info["last_seen"]) > TIMEOUT_LIMIT:
                    info["status"] = "OFFLINE"
                    log(
                        self.node_id,
                        f"ALERT: NODE-{pid} detected OFFLINE (Timeout > {TIMEOUT_LIMIT}s)",
                        Colors.WARNING,
                    )

    def print_status(self):
        if not self.running:
            return
        status_str = f"MyLoad: {self.current_load:.2f} | Peers: "
        for pid, info in self.peer_table.items():
            s_icon = "✅" if info["status"] == "ONLINE" else "❌"
            status_str += f"[N{pid} {s_icon} L:{info['load']:.2f}] "
        print(f"   Node {self.node_id} View: {status_str}")


# --- MAIN DRIVER ---
if __name__ == "__main__":
    nodes = []
    print(
        f"{Colors.HEADER}=== AVVIO SIMULAZIONE UDP HEARTBEAT ({NODES_COUNT} NODI) ==={Colors.ENDC}"
    )
    print(f"Config: Interval={HEARTBEAT_INTERVAL}s, Timeout={TIMEOUT_LIMIT}s\n")

    # 1. Creazione e Avvio Nodi
    for i in range(1, NODES_COUNT + 1):
        n = MockNode(i, NODES_COUNT)
        nodes.append(n)
        n.start()

    try:
        # Loop principale di monitoraggio
        start_time = time.time()
        crashed = False
        recovered = False

        while True:
            time.sleep(2)
            print("\n--- Network Snapshot ---")
            for n in nodes:
                n.print_status()
            print("------------------------")

            elapsed = time.time() - start_time

            # SCENARIO: Uccidi nodo 2 dopo 10 secondi
            if elapsed > 10 and not crashed:
                print(
                    f"\n{Colors.FAIL}!!! SIMULAZIONE CRASH: Uccisione Nodo 2 !!!{Colors.ENDC}\n"
                )
                nodes[1].stop()  # Node 2 è all'indice 1
                crashed = True

            # SCENARIO: Riavvia nodo 2 dopo 25 secondi (15s di downtime)
            if elapsed > 25 and not recovered:
                print(
                    f"\n{Colors.OKBLUE}!!! SIMULAZIONE RECOVERY: Riavvio Nodo 2 !!!{Colors.ENDC}\n"
                )
                # Bisogna ricreare l'oggetto o riaprire la socket, per semplicità ricreiamo
                new_node_2 = MockNode(2, NODES_COUNT)
                nodes[1] = new_node_2
                new_node_2.start()
                recovered = True

            if elapsed > 40:
                print("\nTest Completato. Chiusura.")
                break

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    finally:
        for n in nodes:
            if n.running:
                n.stop()
