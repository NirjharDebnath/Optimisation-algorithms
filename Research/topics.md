## Markov Decision Process (MDP)
---

Markov Decision Process (MDP) is a fundamental concept in reinforcement learning and decision-making under uncertainty. It provides a mathematical framework for modeling decision-making situations where outcomes are partly random and partly under the control of a decision-maker. Here are the key components and characteristics of an MDP:

1. **States (S)**: 
   - The set of all possible situations or configurations in the environment.
   - Example: In a chess game, a state could be the current arrangement of pieces on the board.

2. **Actions (A)**:
   - The set of all possible decisions or moves the agent can make.
   - Example: In chess, actions would be legal moves a player can make.

3. **Transition Probability (P)**:
   - The probability of moving from one state to another when taking a specific action.
   - Represented as P(s'|s,a) - probability of reaching state s' from state s by taking action a.

4. **Rewards (R)**:
   - The immediate feedback or payoff received after taking an action in a given state.
   - Represented as R(s,a,s') - reward received when transitioning from s to s' via action a.

5. **Discount Factor (γ)**:
   - A value between 0 and 1 that determines the importance of future rewards.
   - Lower γ prioritizes immediate rewards, while higher γ values long-term rewards.

Key properties of MDPs:

- **Markov Property**: The future state depends only on the current state and action, not on the history of previous states.
- **Time-homogeneity**: Transition probabilities and rewards do not change over time.

The goal in an MDP is typically to find an optimal policy π*(s) that maximizes the expected cumulative reward:

$$
V^*(s) = \max_{\pi} \mathbb{E} \left[\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t), s_{t+1}) \right]
$$

Understanding MDPs is crucial for implementing reinforcement learning algorithms, as they form the basis for more complex RL frameworks and techniques.

---
