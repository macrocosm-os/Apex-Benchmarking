import asyncio
import yaml
import random

import bittensor as bt

from utils.dataset import DatasetWrapper, create_challenge
from utils.bt_utils import resync_metagraph, get_uids, get_axons, get_incentives, query_miners, process_streams, StreamPromptingSynapse, get_ranks
from utils.logging import init_wandb, log_step, reinit_wandb


async def main():
    # Load the config.yml file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the bittensor objects
    wallet = bt.wallet(name=config.get('wallet_name'), hotkey=config.get('wallet_hotkey'))
    subtensor = bt.subtensor(network=config.get('network'))
    metagraph = subtensor.metagraph(netuid=config.get('netuid'))
    dendrite = bt.dendrite(wallet=wallet)

    # Initialize the dataset
    dataset = DatasetWrapper(name=config.get('dataset_name'), batch_size=config.get('batch_size'), seed=config.get('seed'))

    # Initialize the wandb run
    run = init_wandb(config)

    # Create a loop that runs for the number of epochs specified in the config file
    for epoch in range(config.get('num_epochs')):
        # Reinit the run
        run = reinit_wandb(config, run)

        # Resync the metagraph
        resync_metagraph(metagraph=metagraph, subtensor=subtensor)
        
        # Get the context and challenge
        context = next(dataset)
        challenge = create_challenge(context)

        # Create the synapse
        synapse = StreamPromptingSynapse(roles=["user"], messages=[challenge], task_name = 'multi_choice', seed=random.randint(0, 999999), target_model='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4')

        # Get the uids, axons, and incentives
        uids = get_uids(metagraph=metagraph, uid_sample_size=config.get('uid_sample_size'), percentile=config.get('percentile'))
        axons = get_axons(metagraph, uids)
        incentives = get_incentives(metagraph, uids)
        ranks = get_ranks(metagraph, uids)

        # Query the miners
        streams = await query_miners(dendrite, axons, synapse)

        # Process the streams
        responses = await process_streams(streams)

        # Log the step
        log_step(run=run, context=context, responses=responses, uids=uids, axons=axons, challenge=challenge, incentives=incentives, ranks=ranks)

        # Wait 15 seconds before sending the next query``
        print(f"Completed epoch: {epoch}")
        await asyncio.sleep(15)


if __name__ == "__main__":
    asyncio.run(main())