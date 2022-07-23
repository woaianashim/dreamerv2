import hydra
from time import time
from algo.learner import Plotter


@hydra.main(config_path="configs", config_name="dreamer")
def main(cfg):
    print("Initializing env.")
    env = hydra.utils.instantiate(cfg.Environment)

    print("Initializing logger.")
    logger = hydra.utils.instantiate(cfg.Logger)

    print("Initializing learner.")
    learner = hydra.utils.instantiate(
        cfg.Learner,
        _recursive_=False,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    obs = env.reset()
    learner.reset(obs)

    if cfg.checkpoint != "None":
        print("Loading from checkpoint")
        learner.load(cfg.checkpoint)

    print("Initializing plotter.")
    hydra.utils.instantiate(cfg.Plotter)

    print("Start learning.")
    rewards = 0.0
    episode_steps = 0
    for step in range(1000000):
        if step == cfg.random_steps:
            print("Start using policy.")

        if step < cfg.random_steps:
            action = env.action_space.sample()
        else:
            action = learner(obs)
        obs, reward, done, _ = env.step(action)
        episode_steps += 1

        rewards += reward
        if done:
            obs = env.reset()
            logger.add_episode(step, rewards)
            episode_steps = 0
            rewards = 0.0

        learner.store_step(obs=obs, rew=reward, done=done, action=action)

        if learner.full and step%cfg.update_freq:
            logger.start_log(step)
            loss = learner.learn_step()
            logger.add_loss(step, loss)
            learner.save(step)

        logger.log(step)



if __name__ == "__main__":
    main()
