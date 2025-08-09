## **1. Prediction vs. Control**

### **Prediction**

* **Goal**: Estimate the value function **for a given policy**.
* We are *not* trying to change the policy — just evaluate how good it is.
* We can compute:

  * **State-value function** $V^\pi(s)$ — expected return starting from $s$ following $\pi$.
  * **Action-value function** $Q^\pi(s,a)$ — expected return starting from $s$, taking $a$, then following $\pi$.
* **Example**: TD(0) policy evaluation, Monte Carlo prediction.

---

### **Control**

* **Goal**: Find the **optimal policy** $\pi^*$.
* This means:

  1. Learning $Q^*(s,a)$ or $V^*(s)$ (optimal value functions).
  2. Improving the policy to be greedy w\.r.t. those values.
* **Action-values** $Q(s,a)$ are usually used in control, because they directly tell us which action to choose in each state without needing a model.
* **Example**: SARSA, Q-learning.

---

## **2. Policy Evaluation vs. Policy Iteration**

### **Policy Evaluation**

* Part of **Generalized Policy Iteration** (GPI).
* Means: *Given a fixed policy*, estimate its value function (prediction problem).

### **Policy Iteration**

* An algorithm for control:

  1. **Policy evaluation** — compute $V^\pi$ (prediction).
  2. **Policy improvement** — update $\pi$ to be greedy w\.r.t. $V^\pi$.
  3. Repeat until convergence.
* In practice, modern RL often does this **incrementally**: improve policy while still evaluating (e.g., SARSA, Q-learning).

---

## **Summary**

* **Prediction** = compute the value function (state-value or action-value) for a given policy.
* **Control** = find an optimal policy, usually via learning action-values and improving the policy.
* **Policy evaluation** is *prediction* for a fixed policy.
* **Policy iteration** is a *control* method that alternates prediction (evaluation) and improvement.

---


