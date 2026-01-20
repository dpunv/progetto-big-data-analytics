import json


def create_peers_config(peers, filename="peers.json"):
    try:
        with open(filename, "w") as f:
            json.dump(peers, f, indent=4)
        print(
            f"\nSuccess! Configuration saved to '{filename}' with {len(peers)} peers."
        )
        return True
    except IOError as e:
        print(f"Error saving file: {e}")
        return False


def main():
    print("=== P2P Network Peer Configuration Generator ===")
    try:
        num_peers = int(input("Enter number of peers to configure: "))
    except ValueError:
        print("Invalid number. Exiting.")
        return

    peers = []
    print(
        "\nPlease enter details for each peer (IDs will be auto-assigned starting from 0)."
    )

    for i in range(num_peers):
        print(f"\n--- Peer ID {i} ---")
        url = input("Enter URL/IP (e.g., 127.0.0.1) [default: 127.0.0.1]: ").strip()
        if not url:
            url = "127.0.0.1"

        while True:
            try:
                port_str = input(f"Enter Port for Peer {i}: ")
                port = int(port_str)
                break
            except ValueError:
                print("Invalid port. Please enter a valid integer.")

        peers.append({"id": i, "url": url, "port": port})

    if peers:
        print("Sample content:")
        print(json.dumps(peers[:2], indent=4) + ("\n..." if len(peers) > 2 else ""))
        create_peers_config(peers)


if __name__ == "__main__":
    main()
