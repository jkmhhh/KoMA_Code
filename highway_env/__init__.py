# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


from gymnasium.envs.registration import register


def register_highway_envs():
    """Import the envs module so that envs register themselves."""


    # KoMA_multiroundabout_env.py
    register(
        id='KoMA-merge-roundabout',
        entry_point='highway_env.envs:KoMAMultiRoundAboutEnv'
    )

    # KoMA_Merge_env_generalization.py
    register(
        id='KoMA-merge-generalization',
        entry_point='highway_env.envs:KoMAMergeGeneralizationEnv'
    )

    # KoMA_Merge_onelane_env.py
    register(
        id='KoMA-merge-onelane',
        entry_point='highway_env.envs:KoMAMergeOneLaneEnv'
    )

    # KoMA_Merge_threelane_env.py
    register(
        id='KoMA-merge-threelane',
        entry_point='highway_env.envs:KoMAMergeThreeLaneEnv'
    )
