Trajectory

$$
\{
    s_0,a_0,r_1+V(s_1;\theta_0)-V(s_0;\theta_0), \\
    s_1,a_1,r_2+V(s_2;\theta_1)-V(s_1;\theta_1), \\
    \cdots, \\
    s_{n},a_{n},r_n+V(s_{n+1};\theta_{n})-V(s_{n};\theta_{n}),s_0\}
$$

$$
\begin{aligned}
\sum_r \tilde{r} &= V(s_1;\theta_0) - V(s_0;\theta_0) + V(s_2;\theta_1) - V(s_1;\theta_1) + \cdots + V(s_0;\theta_n) - V(s_n;\theta_n) \\
&= \sum_{i=1}^n \left[V(s_i;\theta_{i-1}) - V(s_i;\theta_i) \right] + V(s_0;\theta_{n}) - V(s_0;\theta_{0}) \\
&= \sum_{i=1}^n \left[V(s_i;\theta_{i-1}) - V(s_i;\theta_i) \right] + \sum_{i=1}^n \left[V(s_0;\theta_i) - V(s_0;\theta_{i-1})\right] \\
&= \sum_{i=1}^n \left[(V(s_i;\theta_{i-1}) - V(s_i;\theta_i)) - (V(s_0;\theta_{i-1}) - V(s_0;\theta_{i})) \right]
\end{aligned}
$$
