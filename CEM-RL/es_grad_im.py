from copy import deepcopy
import argparse

import torch
import pandas as pd

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from ES import sepCEM
from models import Actor, Critic, CriticTD3
from collections import namedtuple
from random_process import GaussianNoise
from memory import Memory
from samplers import IMSampler
from util import *


Sample = namedtuple(
    "Sample", ("params", "score", "gens", "start_pos", "end_pos", "steps")
)
Theta = namedtuple("Theta", ("mu", "cov", "samples"))
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(
    actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False
):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
    """

    if not random:

        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor(state).cpu().data.numpy().flatten()

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -max_action, max_action)

    else:

        def policy(state):
            return env.action_space.sample()

    scores = []
    steps = 0

    for _ in range(n_episodes):

        score = 0
        obs, _ = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_bool = 0 if steps + 1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
    )
    parser.add_argument("--env", default="HalfCheetah-v4", type=str)
    parser.add_argument("--start_steps", default=10000, type=int)

    # DDPG parameters
    parser.add_argument("--actor_lr", default=0.001, type=float)
    parser.add_argument("--critic_lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=1.0, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--layer_norm", dest="layer_norm", action="store_true")

    # TD3 parameters
    parser.add_argument("--use_td3", dest="use_td3", action="store_true")
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument("--gauss_sigma", default=0.1, type=float)

    # OU process parameters
    parser.add_argument("--ou_noise", dest="ou_noise", action="store_true")
    parser.add_argument("--ou_theta", default=0.15, type=float)
    parser.add_argument("--ou_sigma", default=0.2, type=float)
    parser.add_argument("--ou_mu", default=0.0, type=float)

    # ES parameters
    parser.add_argument("--pop_size", default=10, type=int)
    parser.add_argument("--elitism", dest="elitism", action="store_true")
    parser.add_argument("--n_grad", default=5, type=int)
    parser.add_argument("--sigma_init", default=1e-3, type=float)
    parser.add_argument("--damp", default=1e-3, type=float)
    parser.add_argument("--damp_limit", default=1e-5, type=float)
    parser.add_argument("--mult_noise", dest="mult_noise", action="store_true")

    # Sampler parameters
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--k", type=int, default=1)

    # Training parameters
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--max_steps", default=1000000, type=int)
    parser.add_argument("--mem_size", default=1000000, type=int)
    parser.add_argument("--n_noisy", default=0, type=int)

    # Testing parameters
    parser.add_argument("--filename", default="", type=str)
    parser.add_argument("--n_test", default=1, type=int)

    # misc
    parser.add_argument("--output", default="results/", type=str)
    parser.add_argument("--period", default=5000, type=int)
    parser.add_argument("--n_eval", default=10, type=int)
    parser.add_argument(
        "--save_all_models", dest="save_all_models", action="store_true"
    )
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--render", dest="render", action="store_true")

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", "w") as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    actor = Actor(state_dim, action_dim, max_action, args)
    actor_t = Actor(state_dim, action_dim, max_action, args)
    actor_t.load_state_dict(actor.state_dict())
    a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()

    # CEM
    es = sepCEM(
        actor.get_size(),
        mu_init=actor.get_params(),
        sigma_init=args.sigma_init,
        damp=args.damp,
        damp_limit=args.damp_limit,
        pop_size=args.pop_size,
        antithetic=not args.pop_size % 2,
        parents=args.pop_size // 2,
        elitism=args.elitism,
    )
    sampler = IMSampler(es)

    # stuff to save
    df = pd.DataFrame(
        columns=[
            "total_steps",
            "average_score",
            "average_score_rl",
            "average_score_ea",
            "best_score",
        ]
    )

    # training
    step_cpt = 0
    total_steps = 0
    actor_steps = 0
    reused_steps = 0

    es_params = []
    fitness = []
    n_steps = []
    n_start = []

    old_es_params = []
    old_fitness = []
    old_n_steps = []
    old_n_start = []

    while total_steps < args.max_steps:

        fitness = np.zeros(args.pop_size, dtype=np.int64)
        n_start = np.zeros(args.pop_size, dtype=np.int64)
        n_steps = np.zeros(args.pop_size, dtype=np.int64)
        es_params, n_r, idx_r = sampler.ask(args.pop_size, old_es_params)
        print("Reused {} samples".format(n_r))

        # udpate the rl actors and the critic
        if total_steps > args.start_steps:

            for i in range(args.n_grad):

                # set params
                actor.set_params(es_params[i])
                actor_t.set_params(es_params[i])
                actor.optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

                # critic update
                for _ in tqdm(range((actor_steps + reused_steps) // args.n_grad)):
                    critic.update(memory, args.batch_size, actor, critic_t)

                # actor update
                for _ in tqdm(range(actor_steps + reused_steps)):
                    actor.update(memory, args.batch_size, critic, actor_t)

                # get the params back in the population
                es_params[i] = actor.get_params()

        actor_steps = 0
        reused_steps = 0

        # evaluate noisy actor(s)
        for i in range(args.n_noisy):
            actor.set_params(es_params[i])
            f, steps = evaluate(
                actor,
                env,
                memory=memory,
                n_episodes=args.n_episodes,
                render=args.render,
                noise=a_noise,
            )
            actor_steps += steps
            prCyan("Noisy actor {} fitness:{}".format(i, f))

        # evaluate all actors
        for i in range(args.pop_size):

            # evaluate new actors
            if i < args.n_grad or (i >= args.n_grad and (i - args.n_grad) >= n_r):

                actor.set_params(es_params[i])
                pos = memory.get_pos()
                f, steps = evaluate(
                    actor,
                    env,
                    memory=memory,
                    n_episodes=args.n_episodes,
                    render=args.render,
                )
                actor_steps += steps

                # updating arrays
                fitness[i] = f
                n_steps[i] = steps
                n_start[i] = pos

                # print scores
                prLightPurple("Actor {}, fitness:{}".format(i, f))

            # reusing actors
            else:
                idx = idx_r[i - args.n_grad]
                fitness[i] = old_fitness[idx]
                n_steps[i] = old_n_steps[idx]
                n_start[i] = old_n_start[idx]

                # duplicating samples in buffer
                memory.repeat(
                    int(n_start[i]), int((n_start[i] + n_steps[i]) % args.mem_size)
                )

                # adding old_steps
                reused_steps += old_n_steps[idx]

                # print reused score
                prGreen("Actor {}, fitness:{}".format(i, fitness[i]))

        # update ea
        es.tell(es_params, fitness)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # update sampler stuff
        old_fitness = deepcopy(fitness)
        old_n_steps = deepcopy(n_steps)
        old_n_start = deepcopy(n_start)
        old_es_params = deepcopy(es_params)

        # save stuff
        if step_cpt >= args.period:

            # evaluate mean actor over several runs. Memory is not filled
            # and steps are not counted
            actor.set_params(es.mu)
            f_mu, _ = evaluate(
                actor, env, memory=None, n_episodes=args.n_eval, render=args.render
            )
            prRed("Actor Mu Average Fitness:{}".format(f_mu))

            df.to_pickle(args.output + "/log.pkl")
            res = {
                "total_steps": total_steps,
                "average_score": np.mean(fitness),
                "average_score_half": np.mean(
                    np.partition(fitness, args.pop_size // 2 - 1)[args.pop_size // 2 :]
                ),
                "average_score_rl": (
                    np.mean(fitness[: args.n_grad]) if args.n_grad > 0 else None
                ),
                "average_score_ea": np.mean(fitness[args.n_grad :]),
                "best_score": np.max(fitness),
                "mu_score": f_mu,
                "n_reused": n_r,
            }

            if args.save_all_models:
                os.makedirs(
                    args.output + "/{}_steps".format(total_steps), exist_ok=True
                )
                critic.save_model(
                    args.output + "/{}_steps".format(total_steps), "critic"
                )
                actor.set_params(es.mu)
                actor.save_model(
                    args.output + "/{}_steps".format(total_steps), "actor_mu"
                )
            else:
                critic.save_model(args.output, "critic")
                actor.set_params(es.mu)
                actor.save_model(args.output, "actor")
            df.loc[len(df.index)] = pd.Series(res)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)
