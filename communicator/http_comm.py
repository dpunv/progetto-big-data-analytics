import aiohttp

from .base import BaseCommunicator


class HTTPCommunicator(BaseCommunicator):
    """
    Communicator implementation using HTTP protocol.
    """

    async def send(self, ip: str, port: int, query: str, *args):
        """
        Sends an HTTP POST request to a remote peer.
        Serializes complex python objects (sets, tuples) into JSON-compatible lists.
        Deserializes the response back into native types.
        """
        base_url = f"http://{ip}:{port}"

        json_payload = {}
        target_endpoint = f"/{query}"

        # ---------------------------------------------------------------------
        # Request Serialization Logic
        # ---------------------------------------------------------------------

        if query == "get_id":
            # No arguments for get_id
            json_payload = {}

        elif query == "similarity":
            # args[0] is the vector
            json_payload = {"vector": args[0]}

        elif query == "receive":
            # args[0] is vectors_data, args[1] is status
            vectors_data = args[0]
            status = args[1]

            # Serialize vectors: convert sets/tuples to lists for JSON
            serializable_vectors = []
            for v in vectors_data:
                v_list = list(v)

                # v_list[5] is usually the 'destinations' set
                if len(v_list) > 5:
                    if isinstance(v_list[5], (set, frozenset)):
                        v_list[5] = list(v_list[5])

                serializable_vectors.append(v_list)

            json_payload = {"vectors": serializable_vectors, "status": status}

        elif query == "i_am_coord":
            json_payload = {}

        elif query == "set_clusters":
            # args[0] is clusters dict, args[1] is assignment list
            clusters_in = args[0]
            assignment_in = args[1]

            # Serialize clusters dict. Keys must be strings.
            clusters_serializable = {}
            for cid, data in clusters_in.items():
                # Members might have sets (destinations) in them
                ser_members = []
                for m in data["members"]:
                    m_list = list(m)
                    if len(m_list) > 5 and isinstance(m_list[5], (set, frozenset)):
                        m_list[5] = list(m_list[5])
                    ser_members.append(m_list)

                clusters_serializable[str(cid)] = {
                    "center": data["center"],
                    "members": ser_members,
                }

            # Serialize assignment list
            assignment_serializable = []
            for item in assignment_in:
                assignment_serializable.append([item[0], item[1]])

            json_payload = {
                "clusters": clusters_serializable,
                "assignment": assignment_serializable,
            }

        elif query == "search_vectors_local":
            # args[0] is tuples of (values, id)
            # args[1] is top_k
            vecs_in = []
            for v_item in args[0]:
                vecs_in.append({"values": v_item[0], "id": v_item[1]})

            json_payload = {"vectors": vecs_in, "top_k": args[1]}

        elif query == "query":
            # args[0] is input vectors, args[1] is status
            vecs_in = []
            for v_item in args[0]:
                vecs_in.append({"values": v_item[0], "id": v_item[1]})

            json_payload = {"vectors": vecs_in, "status": args[1]}

        elif query == "get_vector_digest":
            json_payload = {}

        elif query == "get_vectors_by_ids":
            # args[0] is list of IDs
            json_payload = {"ids": args[0]}

        elif query == "get_partition_coordinator_id":
            json_payload = {}

        elif query == "respond_to_ping":
            json_payload = {}

        else:
            return {
                "status": -1,
                "error": f"Unknown HTTP query: {query}",
                "response": None,
            }

        async with aiohttp.ClientSession() as session:
            try:
                # ---------------------------------------------------------------------
                # Send Request
                # ---------------------------------------------------------------------
                async with session.post(
                    base_url + target_endpoint, json=json_payload
                ) as response:
                    # Check HTTP status
                    if response.status != 200:
                        text = await response.text()
                        return {
                            "status": -1,
                            "error": f"HTTP {response.status}: {text}",
                            "response": None,
                        }

                    resp_json = await response.json()

                    # ---------------------------------------------------------------------
                    # Response Deserialization Logic
                    # ---------------------------------------------------------------------
                    response_val = resp_json.get("response")

                    if query == "get_vector_digest":
                        # Keys come back as strings, convert to ints, values to tuples
                        new_digest = {}
                        if response_val:
                            for k, v in response_val.items():
                                new_digest[int(k)] = tuple(v)
                        response_val = new_digest

                    elif query == "search_vectors_local" or query == "query":
                        # Convert lists back to tuples
                        if response_val:
                            response_val = [tuple(x) for x in response_val]

                    elif query == "get_vectors_by_ids":
                        # Restore sets and tuples inside vectors
                        if response_val:
                            new_vecs = []
                            for v in response_val:
                                v_list = list(v)
                                if len(v_list) > 5:
                                    # Destinations: list -> frozenset
                                    if v_list[5]:
                                        v_list[5] = frozenset(v_list[5])
                                    else:
                                        v_list[5] = frozenset()

                                # Version: list -> tuple
                                if len(v_list) > 4:
                                    v_list[4] = tuple(v_list[4])

                                new_vecs.append(tuple(v_list))
                            response_val = new_vecs

                    return {"status": 0, "error": None, "response": response_val}

            except Exception as e:
                # Catch connection errors or other exceptions
                return {"status": -1, "error": str(e), "response": None}
