<center>

# Transferring <br> a Well-learned Q-function

</center>

## Problem

If there is an optimal Q-function $Q_{\pi_*}$ on the original MDP $\mathcal{M} = (S, A, P, R, d)$, how to transfer the knowledge in $Q_{\pi_*}$ to the new MDP $\mathcal{M} = (S',A',P,R,d')$ and learn the optimal Q-function $Q_{\pi'_*}$ on the new MDP. Notice that the two MDP have the same transition function and reward function.

## Solution

### State Mapping

$$
\sigma = f_\sigma(s)
$$

### Action Mapping

$$
\alpha = f_\alpha(a)
$$

### Bellman Equation

State mapping and action mapping will produce a new Q-function $Q_{\pi'}(\sigma,\alpha)$ . However, there is still a discrepancy between $Q_{\pi'}$ and the optimal new Q-function $Q_{\pi'_*}$ . Equations (2) are the Bellman equations of $Q_{\pi'}$ and $Q_{\pi'_*}$, and the discrepancy is defined as equation (3).

$$
\begin{aligned}
Q_{\pi'}(\sigma, \alpha) &= Q_{\pi_*}(s, a) = r + \gamma\max_{a'}Q_{\pi_*}(s', a') \\
Q_{\pi'_*}(\sigma, \alpha) &= r + \gamma\max_{\alpha'}Q_{\pi'_*}(\sigma', \alpha') \\
\end{aligned}
\tag{2}
$$

$$
\begin{aligned}
\operatorname{dis}_{\sigma,\alpha}(Q_{\pi'_*}, Q_{\pi'}) &= Q_{\pi'_*}(\sigma, \alpha) - Q_{\pi'}(\sigma, \alpha) \\
&= \gamma\left(
    \max_{a'}Q_{\pi_*}(s', a') - 
    \max_{\alpha'}Q_{\pi'_*}(\sigma', \alpha')
\right)
\end{aligned}
\tag{3}
$$

Therefore the Bellman optimality equation for $Q_{\pi'_*}$ can be derived as equation (4). This equation expressed the value correction process based on the transferred Q-table $Q_={\pi'}$, which can be implemented using value iteration methods.

$$
Q_{\pi'_*}(\sigma, \alpha) = Q_{\pi'}(\sigma, \alpha) + \gamma\left[
    \max_{a'}Q_{\pi_*}(s', a') - 
    \max_{\alpha'}Q_{\pi'_*}(\sigma', \alpha')
\right]
\tag{4}
$$

## Experiment
