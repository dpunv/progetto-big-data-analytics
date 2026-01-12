import subprocess
import time
import requests
import sys
import os
import shutil

# Configuration
NUM_SERVERS = 8
START_PORT = 6333


def generate_compose_and_dirs():
    print(f"Generating compose.yml and creating directories for {NUM_SERVERS} nodes...")

    services = []

    for i in range(NUM_SERVERS):
        http_port = START_PORT + (i * 2)
        grpc_port = http_port + 1
        storage_dir = f"./qdrant_storage_{i}"

        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)

        service_def = f"""  qdrant-{i}:
    image: qdrant/qdrant:latest
    container_name: qdrant-{i}
    ports:
      - "{http_port}:6333"
      - "{grpc_port}:6334"
    volumes:
      - {storage_dir}:/qdrant/storage
    restart: unless-stopped"""
        services.append(service_def)

    compose_content = "services:\n" + "\n".join(services)

    with open("compose.yml", "w") as f:
        f.write(compose_content)

    print("compose.yml generated.")


def cleanup():
    print("Cleaning up...")
    try:
        subprocess.run(["docker", "compose", "down"], check=True)
        print("Docker containers stopped.")
    except Exception as e:
        print(f"Error stopping docker: {e}")

    # Optional: Clean up storage directories?
    # User asked to "clean everything", implies data too?
    # The prompt says "then launch client and finally clean everything".
    # Usually in testing scenarios yes.
    for i in range(NUM_SERVERS):
        storage_dir = f"./qdrant_storage_{i}"
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
    print("Storage directories removed.")


def wait_for_qdrant():
    print("Waiting for Qdrant nodes to be ready...")
    ready_count = 0
    start_time = time.time()

    while ready_count < NUM_SERVERS:
        if time.time() - start_time > 60:
            print("Timeout waiting for Qdrant nodes.")
            return False

        ready_count = 0
        for i in range(NUM_SERVERS):
            port = START_PORT + (i * 2)
            try:
                resp = requests.get(f"http://localhost:{port}/readyz", timeout=1)
                if resp.status_code == 200:
                    ready_count += 1
            except:
                pass

        if ready_count < NUM_SERVERS:
            time.sleep(1)

    print("All Qdrant nodes are ready.")
    return True


def run_simulation():
    print("Starting simulation (client.py)...")
    env = os.environ.copy()
    env["MULTI_NODE_MODE"] = "true"

    try:
        subprocess.run([sys.executable, "client.py"], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("Simulation interrupted.")


def main():
    try:
        generate_compose_and_dirs()

        print("Starting Docker Compose...")
        subprocess.run(["docker", "compose", "up", "-d"], check=True)

        if wait_for_qdrant():
            run_simulation()
        else:
            print("Aborting simulation due to Qdrant startup failure.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
