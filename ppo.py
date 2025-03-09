# adapted from cleanRL
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPO:
    def __init__(self, env,
                 num_minibatches=4,
                 num_steps=256,
                 learning_rate=3e-4,
                 update_epochs=4,
                 clip_coef=0.2,
                 clip_vloss=False,
                 ent_coef=0.0,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 target_kl=None,
                 gamma=0.99,
                 gae_lambda=0.95,
                 norm_adv=True,
                 checkpoint_path=None):
        self.env = env
        self.agent = Agent(env)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-8)

        self.num_steps = num_steps
        self.batch_size = num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_envs = 1

        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.global_step = 0

    def train(self, total_timesteps):
        num_iterations = total_timesteps // self.batch_size
        obs = torch.zeros((self.num_steps, self.num_envs) + self.env.observation_space.shape)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space.shape)
        logprobs = torch.zeros((self.num_steps, self.num_envs))
        rewards = torch.zeros((self.num_steps, self.num_envs))
        dones = torch.zeros((self.num_steps, self.num_envs))
        values = torch.zeros((self.num_steps, self.num_envs))

        for iteration in range(1, num_iterations + 1):
            # TRY NOT TO MODIFY: start the game
            next_obs, _ = self.env.reset()
            next_obs = torch.Tensor(next_obs)
            next_done = torch.zeros(self.num_envs)
            for step in range(0, self.num_steps):
                self.global_step += 1
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.env.step(action)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = reward
                next_obs, next_done = torch.Tensor(next_obs), torch.Tensor([next_done])

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(-1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                       b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

    def eval(self, num_episodes=1, render=False):
        total_rewards = []
        logos = []

        for episode in range(1, num_episodes + 1):
            obs, _ = self.env.reset()
            obs = torch.Tensor(obs)
            episode_reward = 0

            for i in range(8):
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(obs)

                obs, reward, done, _, logos = self.env.step(action)
                episode_reward += reward
                print(reward)

                obs = torch.Tensor(obs)
                if done:
                    break

            if render:
                self.env.render()
                for logo in logos:
                    logo.show()

            total_rewards.append(episode_reward)
            print(f"Episode {episode}: Reward = {episode_reward}")

    def save(self):
        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_{self.global_step}')

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
