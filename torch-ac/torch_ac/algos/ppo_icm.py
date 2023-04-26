import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base_icm import BaseICMAlgo

class PPO_ICM_Algo(BaseICMAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, icm, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, icm_beta=0.5, icm_policy_weight = 1, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, icm, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, icm_beta, icm_policy_weight, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(list(self.acmodel.parameters()) + list(self.icm.parameters()), lr=self.lr)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_forward_loss = []
            log_backward_loss = []
            log_curiosity_loss = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_curiosity_loss = 0
                batch_loss = 0

                batch_fwd_loss = 0
                batch_inv_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    # curiosity loss
                    one_hot_action = F.one_hot(sb.action.to(torch.int64), num_classes=7).float()
                    # Calculate inverse and forward loss
                    action_logits, pred_phi, phi = self.icm(sb.obs, sb.obs_next, one_hot_action)
                    inv_loss = F.cross_entropy(action_logits, one_hot_action)
                    fwd_loss = F.mse_loss(pred_phi, phi) / 2

                    curiosity_loss = (self.icm_beta * fwd_loss + (1 - self.icm_beta)  * inv_loss)

                    curiosity_loss = curiosity_loss.mean()


                    # ac loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # print(loss)

                    total_loss = self.icm_policy_weight * loss + curiosity_loss
                    # total_loss = curiosity_loss
                    
                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_curiosity_loss += curiosity_loss.item()
                    batch_loss += total_loss
                    batch_fwd_loss += fwd_loss.item()
                    batch_inv_loss += inv_loss.item()

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                grad_norm = torch.nn.utils.clip_grad_norm_(list(self.acmodel.parameters()) + list(self.icm.parameters()), self.max_grad_norm)

                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

                log_curiosity_loss.append(batch_curiosity_loss)
                log_forward_loss.append(batch_fwd_loss)
                log_backward_loss.append(batch_inv_loss)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        print("forward loss", numpy.mean(log_forward_loss))
        print("backward loss",numpy.mean(log_backward_loss))
        print("curiosity loss", numpy.mean(log_curiosity_loss))

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
