import argparse
import torch
import sys

from ppo_agent import PPOAgent, PPOConfig
from envs.atari_env import AtariEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt', help='path to checkpoint')
    p.add_argument('--screen_size', type=int, default=84)
    p.add_argument('--render_mode', type=str, default='rgb_array')
    args = p.parse_args()

    ckpt_path = args.ckpt
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_state = ckpt.get('model_state', ckpt)

    # create env to infer shapes
    try:
        env = AtariEnv(env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=args.screen_size, render_mode=args.render_mode)
        in_channels = env.observation_space.shape[2]
        input_size = env.observation_space.shape[0]
        num_actions = env.action_space.n
    except Exception as e:
        print('Warning: failed to create env to infer shapes:', e)
        print('Falling back to defaults: in_channels=4, input_size=84, num_actions from checkpoint if available')
        in_channels = 4
        input_size = args.screen_size
        num_actions = None

    if num_actions is None:
        # try to infer from checkpoint final layer
        # search for actor weight shape [512, num_actions] or [num_actions, ...]
        for k, v in model_state.items():
            if 'actor.weight' in k:
                num_actions = v.shape[0]
                break
        if num_actions is None:
            print('Could not infer num_actions; please supply an env or inspect checkpoint manually.')
            sys.exit(2)

    print(f'Instantiating model with in_channels={in_channels}, num_actions={num_actions}, input_size={input_size}')
    cfg = PPOConfig()
    agent = PPOAgent(in_channels=in_channels, num_actions=num_actions, config=cfg, input_size=input_size)

    current_state = agent.policy.state_dict()

    missing = []
    unexpected = []
    mismatch = []

    for k, v in model_state.items():
        if k not in current_state:
            unexpected.append(k)
        else:
            if v.shape != current_state[k].shape:
                mismatch.append((k, v.shape, current_state[k].shape))

    for k in current_state.keys():
        if k not in model_state:
            missing.append(k)

    if not unexpected and not mismatch and not missing:
        print('Checkpoint is compatible with the current model structure.')
        print('You can resume training or run deterministic eval with this checkpoint.')
        return 0

    print('\nCompatibility report:')
    if unexpected:
        print(f' Unexpected keys in checkpoint (not in model): {len(unexpected)}')
        for k in unexpected[:10]:
            print('  -', k)
    if missing:
        print(f' Missing keys in checkpoint (present in model but not in checkpoint): {len(missing)}')
        for k in missing[:10]:
            print('  -', k)
    if mismatch:
        print(f' Mismatched shapes: {len(mismatch)}')
        for k, s_ckpt, s_model in mismatch[:20]:
            print(f'  - {k}: checkpoint={s_ckpt} vs model={s_model}')

    print('\nIf the model is incompatible, options:')
    print(' - Revert `ppo_agent.py` to the architecture used when checkpoint was created')
    print(' - Create a new model matching checkpoint')
    print(' - Load checkpoint selectively (advanced): load weights for matching keys only')
    return 1


if __name__ == '__main__':
    sys.exit(main())
