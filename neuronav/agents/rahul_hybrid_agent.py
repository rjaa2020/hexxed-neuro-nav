import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.td_agents import QAgent, DynaModule

class HybridAgent(QAgent):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        sarsa_lambda: float = 0.9,
        dyna_num_recall: int = 3,
        dyna_recency: str = "exponential",
        **kwargs
    ):
        super().__init__(
            state_size,
            action_size,
            lr=lr,
            gamma=gamma,
            poltype=poltype,
            beta=beta,
            epsilon=epsilon,
            **kwargs
        )
        self.sarsa_lambda = sarsa_lambda
        self.dyna = DynaModule(state_size, num_recall=dyna_num_recall, recency=dyna_recency)
        self.eligibility_trace = np.zeros((action_size, state_size))

    def q_estimate(self, state):
        return self.Q[:, state]

    def sample_action(self, state):
        return self.base_sample_action(self.q_estimate(state))

    def update_q(self, current_exp, prospective=False):
        s, s_a, s_1, r, d = current_exp
        q_error = self.q_error(s, s_a, s_1, r, d)

        if not prospective:
            self.eligibility_trace[s_a, s] += 1
            self.Q += self.lr * q_error * self.eligibility_trace
            self.eligibility_trace *= self.gamma * self.sarsa_lambda

        return q_error

    def _update(self, current_exp, **kwargs):
        q_error = self.update_q(current_exp, **kwargs)
        self.dyna.update(self, current_exp)
        return q_error

    def reset(self):
        super().reset()
        self.eligibility_trace *= 0.0