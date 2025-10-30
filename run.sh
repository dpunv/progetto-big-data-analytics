#!/bin/bash

# --- Configuration ---
# The number of nodes to run, taken from the first argument or defaulting to 3
N=${1:-3}
# Starting port for FastAPI servers (8001, 8002, ...)
FASTAPI_START_PORT=8000
# Starting port for Qdrant HTTP (6333, 6335, 6337, ...)
QDRANT_START_PORT=6333
# Port step for Qdrant (each instance needs 2 ports)
QDRANT_PORT_STEP=2

# --- PIDs Array ---
# We will store all background server PIDs in this array
PIDS=()

# --- Cleanup Function ---
cleanup() {
    echo ""
    echo "Shutting down..."
    
    # Kill all background server processes
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Stopping ${#PIDS[@]} Python servers (PIDs: ${PIDS[*]})..."
        kill "${PIDS[@]}" 2>/dev/null
    fi
    
    # Stop and remove the docker containers
    echo "Stopping Docker containers..."
    # Use the -f flag to specify the generated compose file
    docker compose -f compose.yml down
    
    # Clean up the generated compose file
    rm -f compose.yml

    for (( i=1; i<=N; i++ )); do
        rm -rf qdrant_storage_$i
    done

    echo "Cleanup complete."
}

# Trap the EXIT and INT (Ctrl+C) signals to run the cleanup function
trap cleanup EXIT INT

# --- 1. Generate docker-compose.yml ---
echo "Generating compose.yml for $N nodes..."
# Start with a clean file
echo "services:" > compose.yml

for (( i=1; i<=N; i++ )); do
    # Calculate ports for this node
    QDRANT_HTTP_PORT=$((QDRANT_START_PORT + (i-1) * QDRANT_PORT_STEP))
    QDRANT_GRPC_PORT=$((QDRANT_HTTP_PORT + 1))
    
    # Append the service definition using a HEREDOC
    cat >> compose.yml << EOL
  qdrant-$i:
    image: qdrant/qdrant:latest
    container_name: qdrant-$i
    ports:
      - "$QDRANT_HTTP_PORT:6333"
      - "$QDRANT_GRPC_PORT:6334"
    volumes:
      - ./qdrant_storage_$i:/qdrant/storage:z
    restart: unless-stopped
EOL
done

echo "compose.yml generated successfully."

# --- 2. Start Docker Containers ---
echo "Starting $N Qdrant databases with Docker Compose..."
# Use the -f flag to specify the generated compose file
docker compose -f compose.yml up -d

echo "Waiting for databases to initialize (10s)..."
sleep 10

# --- 3. Start Python Servers in Background ---
echo "Starting $N Python servers..."

for (( i=1; i<=N; i++ )); do
    # Calculate ports
    NODE_ID="node$i"
    FASTAPI_PORT=$((FASTAPI_START_PORT + i))
    QDRANT_HTTP_PORT=$((QDRANT_START_PORT + (i-1) * QDRANT_PORT_STEP))

    # Start server
    python server.py $NODE_ID $FASTAPI_PORT $QDRANT_HTTP_PORT &
    PID=$! # Store the Process ID
    PIDS+=($PID) # Add PID to our array
    
    echo "Started server '$NODE_ID' on port $FASTAPI_PORT (Qdrant: $QDRANT_HTTP_PORT, PID: $PID)"
done

echo "Waiting for Python servers to start (5s)..."
sleep 5

# --- 4. Run Main Application ---
echo "========================================="
echo "Running main application (qdrant_app.py) for $N nodes..."
echo "========================================="
# Pass the number of nodes (N) as an argument to the app
python qdrant_app.py $N

echo "========================================="
echo "Application finished."
echo "========================================="

# The 'trap' will handle the cleanup automatically
exit 0