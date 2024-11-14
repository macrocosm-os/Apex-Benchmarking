import bittensor as bt
from cachetools import TTLCache
from functools import wraps
import pydantic
from typing import List, AsyncIterator
from starlette.responses import StreamingResponse
import time
import numpy as np
from typing import Awaitable

cache = TTLCache(maxsize=1, ttl=660)  # 11 minutes = 660 seconds

def only_if_time_passed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'resync' not in cache:
            result = func(*args, **kwargs)
            cache['resync'] = time.time()
            return result
        return None
    return wrapper

@only_if_time_passed
def resync_metagraph(metagraph = None, subtensor = None):
    metagraph.sync(subtensor=subtensor)

def get_uids(metagraph = None, uid_sample_size: int = 5, percentile: int = 50):

    percentile_50 = np.percentile(metagraph.I, percentile)
    uids = np.where((metagraph.S < 100) & (metagraph.I >= percentile_50))[0]
    is_serving_mask = np.array([metagraph.axons[uid].is_serving for uid in uids])
    serving_uids = uids[is_serving_mask]

    # Use the filtered indices to select uid_sample_size number of filtered indices
    selected_uids = np.random.choice(serving_uids, uid_sample_size, replace=False)

    return selected_uids

def get_axons(metagraph = None, uids: np.ndarray = None):
    axons = [metagraph.axons[uid] for uid in uids]
    return axons

def get_incentives(metagraph = None, uids: np.ndarray = None):
    incentives = [metagraph.I[uid] for uid in uids]
    return incentives

def get_ranks(metagraph = None, uids: np.ndarray = None):
    incentives = [metagraph.I[uid] for uid in uids]
    sorted_uids = [uid for _, uid in sorted(zip(incentives, uids), reverse=True)]
    ranks = [sorted_uids.index(uid) + 1 for uid in uids]
    return ranks

async def query_miners(dendrite, axons, synapse):
    streams = await dendrite(
        axons,
        synapse,
        deserialize=False,
        timeout=15,
        streaming=True,
    )
    return streams

async def process_streams(streams:list[Awaitable]):
    responses = []
    for stream in streams:
        async for chunk in stream:
            pass
        responses.append(chunk) # The last chunk contains the synapse
    return responses

# Direct copy of the StreamPromptingSynapse class from the prompting/protocol.py file
class StreamPromptingSynapse(bt.StreamingSynapse):
    roles: List[str] = pydantic.Field(..., title="Roles", description="A list of roles. Immutable.", allow_mutation=False)
    messages: List[str] = pydantic.Field(..., title="Messages", description="A list of messages. Immutable.", allow_mutation=False)
    required_hash_fields: List[str] = pydantic.Field(["messages"], title="Required Hash Fields", allow_mutation=False)
    completion: str = pydantic.Field("", title="Completion", description="Completion status of the object.")
    task_name: str = pydantic.Field(..., title="Task", description="The task for the current StreamPromptingSynapse object. This attribute is immutable.", allow_mutation=False)
    target_model: str | None = pydantic.Field(None, title="Target Model", description="The model the miner should use for generations. If none, the miner should respond with whatever he thinks is best.", allow_mutation=False)
    seed: int | None = pydantic.Field(None, title="Seed", description="The seed for that the miner must use for generations. This is only used in combination with the target_model.", allow_mutation=False)

    async def process_streaming_response(self, response: StreamingResponse) -> AsyncIterator[str]:
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            self.completion += tokens
            yield tokens

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
        headers = {k.decode("utf-8"): v.decode("utf-8") for k, v in response.__dict__["_raw_headers"]}
        extract_info = lambda prefix: {key.split("_")[-1]: value for key, value in headers.items() if key.startswith(prefix)}
        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "roles": self.roles,
            "messages": self.messages,
            "completion": self.completion,
        }