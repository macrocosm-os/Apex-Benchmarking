import wandb
from datetime import datetime, timedelta, timezone


def init_wandb(config):
    # Extract the wandb token and remove it from the config to avoid logging it
    wandb_token = config.pop('wandb_token', None)
    
    # Log in to Wandb using the access token
    if wandb_token and not config.get('wandb_offline', False):
        wandb.login(key=wandb_token, relogin=True)
    
    tags = ['miner', config.get('dataset_name')]

    mode = "offline" if config.get('wandb_offline', False) else "online"
    # Initialize a Wandb run with the given entity and project name
    run = wandb.init(
        entity=config.get('entity'),
        project=config.get('project'),
        config=config,
        mode = mode,
        tags=tags, 
        reinit=True
    )
    return run

def reinit_wandb(config, run):
    # Check if the run has been alive for more than hours specified in config, reinit if it has
    if datetime.now(timezone.utc) - datetime.fromtimestamp(run.start_time, tz = timezone.utc) > timedelta(hours=config.get('wandb_max_run_length', 24)):
        run.finish()
        return init_wandb(config)
    else:
        return run

def log_step(run, context: dict, responses: list, uids: list, axons: list, challenge: str, incentives: list, ranks: list):

    run.log({
        'topic': context['topic'],
        'query': context['query'], 
        'options': context['options'],
        'answer_idx': context['answer_idx'],
        'answer': context['answer'],
        'completions': [synapse.completion for synapse in responses],
        'uids': uids,
        'axons': [axon.to_parameter_dict() for axon in axons],
        'challenge': challenge,
        'incentives':incentives,
        'ranks': ranks,
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )