# Booster Landing RL: From MDP Fundamentals to the Current CUDA Training Loop

> Working-tree snapshot: 26 July 2026
> Primary subject: `booster_landing_continuous` using PufferLib's native CUDA
> backend
> Intended reader: familiar with basic MDPs and value functions, but not
> necessarily policy gradients, PPO, recurrent networks, or this codebase

This document explains the system that is actually present in this working
tree. It is not a generic description of how PPO might work. In particular, it
covers the current C environment, the native Gaussian policy, the native
MinGRU, PufferLib's precise rollout layout, its PPO/GAE variation, the Muon
optimizer, the recurrent-state workaround in the environment, and the
experiments and checkpoints currently on disk.

The repository also contains the older discrete environment
`ocean/booster_landing`. That environment supplied many of the lessons used by
the continuous version, so its successful training history is summarized in
the final appendix. The main body describes the continuous environment because
that is the active Booster Landing line of work.

The most important source files are:

- [`env.c`](./env.c): resets, observations, physics, rewards, termination, and
  episode metrics.
- [`booster_landing_continuous.h`](./booster_landing_continuous.h): environment
  state and log structures.
- [`binding.c`](./binding.c): exposes the C environment to the native vector
  runner and declares two continuous action heads.
- [`config/booster_landing_continuous.ini`](../../config/booster_landing_continuous.ini):
  the checked-in experiment defaults.
- [`scripts/train_booster_continuous.py`](../../scripts/train_booster_continuous.py):
  the temporary native-cleanup workaround used to launch training.
- [`src/models.cu`](../../src/models.cu): native encoder, MinGRU, decoder, and
  their backward passes.
- [`src/pufferlib.cu`](../../src/pufferlib.cu): action sampling, rollout
  buffers, GAE/V-trace, PPO, and training orchestration on the GPU.
- [`src/muon.cu`](../../src/muon.cu): the optimizer that updates the parameters.
- [`src/vecenv.h`](../../src/vecenv.h): CPU environment parallelism and
  CPU/GPU transfers.
- [`experiments/booster_landing.md`](../../experiments/booster_landing.md): the
  detailed chronological experiment journal.

## 1. The whole system in one picture

At one instant, one booster environment contains physical variables such as
altitude, vertical speed, horizontal position, tilt, angular velocity, and
fuel. The environment turns those variables into eight normalized numbers.
The neural network reads those eight numbers plus its recurrent memory and
outputs:

1. the mean of a Gaussian for main-engine control;
2. the mean of a Gaussian for attitude control; and
3. an estimate of expected future discounted reward.

Two learned, state-independent log standard deviations complete the two
Gaussian distributions. PufferLib samples one raw action from each Gaussian.
The environment applies a `tanh` squash, converts the normalized commands into
throttle and signed side thrust, advances the physics by 50 ms, and emits a
reward, a terminal flag, and the next observation.

That loop runs in thousands of environments at once:

```text
physical state s_t
      |
      v
8-number observation o_t ------ previous MinGRU memory m_{t-1}
      |                                      |
      +--------------> neural network <------+
                            |
                 +----------+----------+
                 |                     |
        Gaussian policy             value V_t
           mu_t, sigma
                 |
          sample raw a_t
                 |
        clamp/map to actuators
                 |
          C physics step
                 |
      r_{t+1}, done_{t+1}, o_{t+1}
                 |
        store a rollout of 128 steps
                 |
         GAE targets and advantages
                 |
        PPO actor + critic losses
                 |
     backpropagation through MinGRU
                 |
       gradient clipping + Muon
                 |
         updated network weights
```

There is no gradient through the C physics. The environment is treated as a
black-box source of experience. The mathematical trick that makes learning
possible without differentiating through physics is the policy-gradient
identity, which uses the log probability of sampled actions. We will build up
to that carefully.

## 2. RL foundations in the language of this task

### 2.1 Agent, environment, and trajectory

The **agent** is the neural-network policy. The **environment** is the booster
simulator. At time step \(t\):

1. the environment has state \(s_t\);
2. the agent receives observation \(o_t\);
3. the agent samples action \(a_t\);
4. the environment produces the next state \(s_{t+1}\), reward \(r_{t+1}\),
   and terminal flag \(d_{t+1}\).

A complete episode is a trajectory

$$
\tau =
(s_0,o_0,a_0,r_1,s_1,o_1,a_1,r_2,\ldots,s_T).
$$

The indexing above is the usual RL convention: action \(a_t\) causes reward
\(r_{t+1}\). This convention matters later because the native rollout buffer
also stores the newly arrived reward alongside the newly arrived observation.

### 2.2 The Markov property, state, and observation

In an ideal Markov decision process, the current state contains everything
needed to determine the distribution of the next state:

$$
p(s_{t+1},r_{t+1}\mid s_0,a_0,\ldots,s_t,a_t)
=
p(s_{t+1},r_{t+1}\mid s_t,a_t).
$$

The environment's physical state is approximately

$$
s_t^\text{physical}
=
(y_t,v_t,x_t,v^x_t,\theta_t,\omega_t,f_t),
$$

where:

- \(y\) is altitude;
- \(v\) is vertical velocity, positive upward;
- \(x\) is horizontal displacement from the pad center;
- \(v^x\) is horizontal velocity;
- \(\theta\) is booster angle;
- \(\omega\) is angular velocity; and
- \(f\) is remaining fuel.

For the *exact implementation*, state also includes the episode tick, the
fixed reflection flag, the rollout phase, whether a completed episode is
waiting for a rollout boundary, and the reset random-number-generator state.
The tick matters because timeout occurs after a fixed number of steps.

The policy does not see all of that state. It receives an eight-dimensional
observation. In particular, it does not see the tick. Two physically identical
observations at ticks 10 and 1199 can have different termination prospects.
Clipping can also map multiple physical states to the same observation.
Strictly speaking, the agent therefore faces a small **partially observable
MDP**, or POMDP, rather than a perfectly observed MDP.

That is one reason a recurrent network is reasonable: it can use recent
history to form an internal information state. In this environment, most
important physical variables are already observed, so recurrence is more of a
temporal feature extractor and partial-observability aid than the only source
of state information.

### 2.3 Reward and return

The reward is not the objective by itself. The objective is the discounted sum
of future rewards, called the **return**:

$$
G_t
=
\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}.
$$

The current training discount is

$$
\gamma = 0.9999.
$$

A reward 500 simulator steps in the future is discounted by
\(0.9999^{500}\approx0.951\), so discounting alone barely reduces it. However,
the GAE estimator used for policy credit includes an additional factor
\(\lambda=0.97\), which makes direct advantage credit decay much faster. That
distinction is important.

The policy objective is

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}[G_0],
$$

where \(\theta\) denotes the neural-network parameters and the expectation is
over random resets and stochastic actions.

### 2.4 Policy, value, Q-value, and advantage

A stochastic policy describes a distribution over actions:

$$
\pi_\theta(a_t\mid I_t),
$$

where \(I_t\) is the information available to the policy. For this recurrent
agent,

$$
I_t=(o_t,m_{t-1}),
$$

with \(m_{t-1}\) the MinGRU recurrent state.

The **state value** under the current policy is

$$
V^{\pi}(I_t)
=
\mathbb{E}_{\pi}\left[G_t\mid I_t\right].
$$

The **action value** would be

$$
Q^{\pi}(I_t,a_t)
=
\mathbb{E}_{\pi}\left[G_t\mid I_t,a_t\right].
$$

The **advantage**

$$
A^\pi(I_t,a_t)
=
Q^\pi(I_t,a_t)-V^\pi(I_t)
$$

answers the useful learning question:

> Was the sampled action better or worse than the action we normally expect to
> take from this situation?

A positive advantage says “make this sampled action more likely in similar
conditions.” A negative advantage says “make it less likely.”

The Booster Landing network directly outputs an approximation to \(V^\pi\).
It does **not** output a Q-function. PufferLib estimates advantages afterward
from rewards and successive value predictions.

### 2.5 Actor-critic

This is an **actor-critic** system:

- The **actor** is the Gaussian policy. It chooses actions.
- The **critic** is the scalar value output. It predicts the future return.

They share the encoder and MinGRU. Only the final decoder rows differ. This is
parameter-efficient, and the critic can help shape useful shared features, but
it also means actor and critic gradients can interact in the shared network.

## 3. What exactly is the neural network approximating?

This deserves a direct answer.

For one observation and recurrent state, the network implements a function

$$
f_\theta(o_t,m_{t-1})
=
(\mu_{t,0},\mu_{t,1},\widehat V_t,m_t).
$$

There is also a learned vector

$$
\ell=(\ell_0,\ell_1),
\qquad
\sigma_i=e^{\ell_i},
$$

that does not depend on the observation. The policy is

$$
\pi_\theta(a_t\mid I_t)
=
\mathcal N(a_{t,0};\mu_{t,0},\sigma_0^2)
\mathcal N(a_{t,1};\mu_{t,1},\sigma_1^2),
$$

where the juxtaposition above means a product of two independent densities.
Equivalently, its joint log probability is the sum of the two log
probabilities.

So the network learns two related functions:

1. a map from recent observed flight history to a distribution over the next
   raw control action; and
2. a map from recent observed flight history to the expected discounted return
   if the current policy is followed.

It is **not** learning:

- a model of the next physical state;
- the equations of motion explicitly;
- the reward function;
- a Q-table;
- the analytically optimal action;
- the deterministic controller in the benchmark; or
- a supervised target action supplied by a teacher.

The actor becomes a controller indirectly. Actions that participate in
better-than-expected trajectories get higher probability; actions that
participate in worse-than-expected trajectories get lower probability.

The critic's target is the return formed from the rewards that the native
trainer actually uses. The environment clamps each final transition reward to
\([-1,1]\), and the native trainer applies the same clamp again. The logged
`episode_return` and the critic therefore use the same reward stream.

## 4. The Booster Landing decision process

### 4.1 Units and sign conventions

The simulator uses:

| Quantity | Symbol | Unit | Convention |
|---|---:|---:|---|
| Altitude | \(y\) | m | Ground is \(y=0\) |
| Vertical velocity | \(v\) | m/s | Up is positive; descent is negative |
| Horizontal position | \(x\) | m | Pad center is \(x=0\) |
| Horizontal velocity | \(v^x\) | m/s | Sign follows \(x\) |
| Angle | \(\theta\) | rad | Zero is upright |
| Angular velocity | \(\omega\) | rad/s | Sign follows angle |
| Fuel | \(f\) | kg | Starts at 100 kg |
| Time step | \(\Delta t\) | s | 0.05 s, or 20 Hz |

Each episode can run for 1200 simulator steps, which is 60 seconds of simulated
time.

### 4.2 Reset distribution

The checked-in continuous config currently resets from:

| Variable | Checked-in distribution |
|---|---:|
| Altitude | uniform from 100 to 500 m |
| Downward speed | uniform from 2 to 15 m/s, stored as negative \(v\) |
| Horizontal position | uniform from -40 to 40 m |
| Horizontal velocity | exactly 0 m/s |
| Angle | uniform from -0.05 to 0.05 rad |
| Angular velocity | exactly 0 rad/s |
| Fuel | exactly 100 kg |

All listed random variables are sampled independently by each C environment.
Each native environment has its own `rand_r` state derived from its environment
index plus `reset_seed`.

Two config fields can be misleading:

- `initial_altitude = 2000` is currently the altitude observation's
  normalization scale and a rendering scale. It is not the fixed reset
  altitude.
- `initial_downward_velocity = 35` is retained in the environment structure,
  but the active reset path uses the configured reset range instead.

The intended full benchmark is substantially harder: altitude through 1500 m,
downward speed 5–25 m/s, horizontal velocity through \(\pm5\) m/s, angular
velocity through \(\pm0.05\) rad/s, and stricter touchdown limits. The current
checked-in reset is a curriculum, not the final task.

### 4.3 Observation space

The observation is a vector \(o_t\in\mathbb R^8\):

| Index | Meaning | Formula |
|---:|---|---|
| 0 | normalized altitude | \(\operatorname{clip}(y/2000,0,1)\) |
| 1 | normalized vertical velocity | \(\operatorname{clip}(v/200,-1,1)\) |
| 2 | normalized horizontal position | \(\operatorname{clip}(\rho x/420,-1,1)\) |
| 3 | normalized horizontal velocity | \(\operatorname{clip}(\rho v^x/40,-1,1)\) |
| 4 | normalized angle | \(\operatorname{clip}(\rho\theta/\pi,-1,1)\) |
| 5 | normalized angular velocity | \(\operatorname{clip}(\rho\omega/1,-1,1)\) |
| 6 | fuel fraction | \(\operatorname{clip}(f/100,0,1)\) |
| 7 | landing-safety risk | \(\max(R_\text{stop},R_\text{fuel})\) |

Here \(\rho\in\{-1,+1\}\) is the horizontal canonicalization frame.

Observation 7 is a hand-engineered feature computed from the known physics.
Providing it does not force the network to follow a controller. It gives the
network an easier scalar summary of whether its current speed and fuel make a
safe vertical landing viable. The policy can still ignore or reinterpret it.

The observation does not include:

- the episode tick or remaining time;
- the previous action;
- an explicit mass value, although mass is determined by observed fuel;
- acceleration;
- a target coordinate, because the pad is always centered at zero; or
- separate stopping and fuel risks—the observation contains their maximum.

### 4.4 Horizontal canonicalization

At reset, if canonicalization is enabled and the **initial** horizontal
velocity is positive, the episode sets \(\rho=-1\); otherwise \(\rho=+1\).
That frame remains fixed for the whole episode.

The environment multiplies \(x\), \(v^x\), \(\theta\), and \(\omega\) by
\(\rho\) before presenting them to the policy, then multiplies the policy's
signed attitude command by \(\rho\) on the way back into physical coordinates.
This maps mirror-image initial-velocity cases to a common policy frame.

This is not a time-varying “always make velocity negative” transformation. It
is one reflection chosen at reset. In the current checked-in curriculum,
initial horizontal velocity is exactly zero, so the condition
`start_x_velocity > 0` is never true and canonicalization is dormant. It
becomes active again when nonzero horizontal-velocity starts are restored.

### 4.5 Raw action space and physical action space

The binding declares two heads with action sizes `{1, 1}`. In the native
backend, a head size of one denotes a continuous dimension. The learned raw
action is therefore

$$
a_t=(a^\text{main}_t,a^\text{att}_t)\in\mathbb R^2.
$$

It is important that the raw mathematical action space is unbounded: PufferLib
samples ordinary Gaussians. The environment maps each coordinate through
`tanh`:

$$
u^\text{main}=\tanh(a^\text{main}),
\qquad
u^\text{att}=\tanh(a^\text{att}).
$$

The normalized commands are strictly inside \((-1,1)\) for every finite raw
sample. For the unidirectional main engine, the normalized command is
affinely rescaled:

$$
q
=
\frac{u^\text{main}+1}{2}.
$$

Thus:

- increasingly negative raw actions approach zero throttle;
- raw action \(0\) means 50% throttle; and
- increasingly positive raw actions approach full throttle.

The zero-centered Gaussian now starts at the midpoint of the physical throttle
range instead of placing half its probability mass in an identical coast
region.

For attitude control:

$$
c_\text{policy}
=
u^\text{att},
\qquad
c=\rho c_\text{policy}.
$$

The command \(c\) is signed side-thrust authority. Positive and negative
commands are opposite directions. Unlike the old three-binary-action
environment, the continuous environment cannot command both side thrusters at
once.

PufferLib stores the raw Gaussian sample and computes its raw Gaussian log
probability. This remains valid for the PPO likelihood ratio because `tanh`
and the throttle rescaling are fixed one-to-one transformations: their
Jacobian factor is identical under the old and new policies and cancels from
the ratio. The logged Gaussian entropy is still raw-space entropy rather than
actuator-space entropy; the current entropy coefficient is zero.

Unlike clipping, the squash has no finite interval on which different raw
samples produce exactly the same actuator command. This removes the
unbounded coast and full-thrust plateaus that trapped the previous policy.

### 4.6 Fuel use and low-fuel scaling

At each 50 ms step, requested fuel is

$$
\Delta f_\text{main}
=
6(0.05)q
=
0.3q\ \text{kg},
$$

$$
\Delta f_\text{side}
=
0.8(0.05)|c|
=
0.04|c|\ \text{kg}.
$$

If there is insufficient fuel for both requests, the environment computes

$$
\eta
=
\min\left(
\frac{f}{\Delta f_\text{main}+\Delta f_\text{side}},
1
\right)
$$

and scales both commands by the same \(\eta\). This preserves the requested
main/side proportion while exhausting the remaining fuel exactly.

The mass used for that step's acceleration is the pre-burn mass

$$
m=120+f.
$$

At full fuel, \(m=220\) kg. A continuously full main engine consumes all 100 kg
in about \(100/6=16.67\) seconds.

### 4.7 Equations of motion

Let \(q_e=\eta q\) and \(c_e=\eta c\). The main, side, and angular
accelerations are

$$
a_\text{main}=\frac{3300q_e}{m},
$$

$$
a_\text{side}=\frac{80c_e}{m},
$$

$$
\alpha
=
\frac{80(8)c_e}{3200}
=
0.2c_e\ \text{rad/s}^2.
$$

The world-frame accelerations are

$$
a_y=a_\text{main}\cos\theta-9.81,
$$

$$
a_x=a_\text{main}\sin\theta+a_\text{side}.
$$

The environment uses semi-implicit Euler integration:

$$
v_{t+1}=v_t+a_y\Delta t,
$$

$$
v^x_{t+1}=v^x_t+a_x\Delta t,
$$

$$
\omega_{t+1}
=
\operatorname{clip}\left(
0.99(\omega_t+\alpha\Delta t),-1,1
\right),
$$

$$
y_{t+1}=y_t+v_{t+1}\Delta t,
$$

$$
x_{t+1}=x_t+v^x_{t+1}\Delta t,
$$

$$
\theta_{t+1}=\theta_t+\omega_{t+1}\Delta t.
$$

Velocity is updated before position, which is why this is semi-implicit rather
than fully explicit Euler.

Some useful physical scales:

- At the initial 220 kg mass, full thrust gives
  \(3300/220=15\ \text{m/s}^2\) gross vertical acceleration.
- Upright net acceleration at that mass is about
  \(15-9.81=5.19\ \text{m/s}^2\) upward.
- Full side thrust at that mass gives only about
  \(80/220=0.364\ \text{m/s}^2\) lateral acceleration.
- Full attitude command gives \(0.2\ \text{rad/s}^2\) angular acceleration
  before damping.

These numbers explain why vertical planning, fuel timing, and early horizontal
correction are important. The booster cannot cheaply erase a large lateral
error at the last moment.

### 4.8 Terminal conditions and success

An episode ends if any of these occurs:

1. the booster reaches the ground, \(y\leq0\);
2. it leaves the horizontal world, \(|x|>420\);
3. it has effectively no fuel and even ballistic free fall would exceed the
   allowed vertical landing speed; or
4. it reaches 1200 steps.

The fuel-unrecoverable test uses

$$
v_\text{ballistic}
=
\sqrt{v^2+2g\max(y,0)}
$$

and terminates if fuel is empty and this is above the vertical touchdown
limit.

Touching the ground counts as success only when all of these are true:

$$
|x|\leq \frac{160}{2}=80\ \text{m},
$$

$$
|v|\leq v_\text{land,max},
\quad
|v^x|\leq v^x_\text{land,max},
$$

$$
|\theta|\leq\theta_\text{max},
\quad
|\omega|\leq\omega_\text{max}.
$$

The checked-in curriculum limits are:

| Criterion | Current checked-in limit | Intended strict target |
|---|---:|---:|
| Vertical speed | 8 m/s | 5 m/s |
| Horizontal speed | 2 m/s | 1 m/s |
| Angle | 0.06 rad | 0.03 rad |
| Angular speed | 0.10 rad/s | 0.05 rad/s |

Failure metrics overlap. For example, one crash can fail horizontal speed,
angle, and fuel criteria simultaneously. Failure percentages should not be
expected to sum to the total failure rate.

## 5. Reward shaping

### 5.1 Terminal objective

At a terminal transition, the environment adds

$$
r_\text{terminal}
=
\begin{cases}
+1,&\text{successful landing},\\
-1,&\text{any failure}.
\end{cases}
$$

This binary outcome is the mission score. Dense shaping is added to make
credit assignment easier.

### 5.2 Stopping-distance risk

Define downward speed

$$
d=\max(-v,0)
$$

and available upright component of full thrust

$$
T_y=3300\max(\cos\theta,0).
$$

The estimated braking acceleration at the current mass is

$$
b=\frac{T_y}{120+f}-g.
$$

If \(d\) already satisfies the landing-speed limit, required stopping distance
is zero. Otherwise,

$$
D_\text{stop}
=
\frac{d^2-v_\text{land,max}^2}{2b}
$$

when \(b>0.001\). If there is effectively no braking authority, risk is set to
one. The normalized risk is

$$
R_\text{stop}
=
\operatorname{clip}
\left(
\frac{D_\text{stop}}{\max(y,1)},0,1
\right).
$$

A value near one means the estimated stopping distance consumes all available
altitude.

### 5.3 Required-fuel risk

The fuel estimate assumes a coast-then-burn maneuver and approximates thrust
using the expected mid-burn mass

$$
m_\text{mid}=120+\frac{f}{2}.
$$

Let

$$
b_\text{mid}
=
\frac{T_y}{m_\text{mid}}-g.
$$

The estimated switch speed is

$$
d_\text{switch}
=
\sqrt{
\frac{
2y+v^2/g+v_\text{land,max}^2/b_\text{mid}
}{
1/g+1/b_\text{mid}
}
},
$$

clamped to be at least the current downward speed. The estimated burn time and
fuel are

$$
t_\text{burn}
=
\frac{\max(d_\text{switch}-v_\text{land,max},0)}
{b_\text{mid}},
$$

$$
\widehat f_\text{required}=6t_\text{burn}.
$$

Finally,

$$
R_\text{fuel}
=
\operatorname{clip}
\left(
\frac{\widehat f_\text{required}}{\max(f,0.01)},0,1
\right).
$$

This is a viability estimate, not a generic fuel-economy reward. It asks
whether the fuel remaining appears sufficient for a safe vertical braking
plan.

Observation 7 is

$$
o_7=\max(R_\text{stop},R_\text{fuel}).
$$

### 5.4 Current landing-safety potential

The continuous environment now uses the exact potential that supported the
successful discrete curriculum:

$$
\Phi(s)
=
-\max(R_\text{stop},R_\text{fuel}).
$$

Observation 7 and the shaping potential therefore use the same safety-risk
quantity. Horizontal position, horizontal speed, angle, and angular velocity
remain terminal success criteria and observations, but they do not have
separate dense reward terms.

### 5.5 Potential-based shaping

For a nonterminal transition, the shaping reward is

$$
F(s_t,s_{t+1})
=
0.1\left[
\gamma\Phi(s_{t+1})-\Phi(s_t)
\right],
$$

with the same \(\gamma=0.9999\) used by training. At a terminal state, the
implementation defines \(\Phi(s_{t+1})=0\).

Why use this form instead of simply rewarding low risk? Consider the
discounted sum of shaping terms:

$$
\begin{aligned}
\sum_{t=0}^{T-1}\gamma^tF_t
&=
0.1\sum_{t=0}^{T-1}
\gamma^t
\left[
\gamma\Phi(s_{t+1})-\Phi(s_t)
\right]\\
&=
0.1\left[
-\Phi(s_0)+\gamma^T\Phi(s_T)
\right].
\end{aligned}
$$

All intermediate terms cancel. If terminal potential is zero, the shaping
changes the episode return only by \(-0.1\Phi(s_0)\), which is fixed for a
given starting state. In the textbook setting, this gives denser feedback
without changing which policy is optimal.

Intuitively:

- moving from a dangerous state to a safer state gives positive shaping;
- moving toward worse vertical landing viability gives negative shaping;
- lingering at a fixed negative potential gives a tiny positive shaping term
  because \((\gamma-1)\Phi>0\); this is part of the telescoping construction,
  not a separate survival objective; and
- termination cancels the remaining potential.

### 5.6 Current reward-clipping caveat

After combining shaping and terminal reward, the environment applies

$$
\bar r=\operatorname{clip}(r,-1,1)
$$

and logs that clamped value. The native trainer applies the same clamp before
computing advantages, which does not change the reward a second time.

This has two consequences:

1. The potential-invariance proof no longer applies exactly whenever the raw
   combined reward exceeds the clamp.
2. A successful transition with \(+1\) terminal reward plus positive potential
   cancellation can be clipped to \(+1\), discarding part of the shaping.

This matches the reward implementation used by the successful discrete
curriculum. `score` remains the clearest mission metric.

## 6. The exact native neural network

### 6.1 Native versus Python model selection

Normal Booster training uses the compiled `_C` backend. In this path, the
policy is hard-wired in `src/pufferlib.cu` to:

```text
linear encoder -> MinGRU -> linear actor/value decoder
```

The `[torch] network = MinGRU` config entry matters to the slow Python/Torch
backend selected with `--slowly`; it is not what dynamically chooses the
native architecture. The native code constructs MinGRU directly.

### 6.2 Shapes and parameter count

With eight observations, hidden size 128, one MinGRU layer, and two continuous
actions:

| Component | Parameter shape | Count |
|---|---:|---:|
| Linear encoder | \(128\times8\) | 1,024 |
| MinGRU projection | \(384\times128\) | 49,152 |
| Actor/value decoder | \(3\times128\) | 384 |
| Global action log standard deviations | \(2\) | 2 |
| **Nominal logical total** |  | **50,562** |

There are no biases in these linear maps. The decoder has three rows: two
policy means and one value. A continuous checkpoint contains 50,562 FP32
slots, or \(50{,}562\times4=202{,}248\) bytes. A source-level alignment issue
means those slots are not currently a perfectly packed one-to-one
serialization of the nominal parameters; Section 17 explains the distinction.

The active extension was built in BF16 mode:

- rollout/training parameters and many activations use BF16;
- an FP32 master copy is maintained for optimizer updates;
- saved checkpoints are flat FP32 weights.

### 6.3 Encoder

The encoder is simply

$$
x_t=W_eo_t,
\qquad
W_e\in\mathbb R^{128\times8}.
$$

There is no bias and no activation after this map. Nonlinearity and temporal
processing occur in the MinGRU.

## 7. MinGRU from first principles

### 7.1 Why recurrence?

A feed-forward policy would compute the same output whenever it sees the same
observation. A recurrent policy also has memory:

$$
m_t=g_\theta(x_t,m_{t-1}).
$$

This lets it distinguish situations that look similar now but arrived through
different recent histories. For Booster Landing, possible uses include:

- estimating trends from noisy or clipped observations;
- remembering its recent control pattern;
- interpreting the hidden episode time;
- smoothing a control strategy over time; and
- carrying information not explicit in one observation.

The current hidden state is reset every 128-step rollout rather than only at
true episode boundaries. Its effective recurrent context is therefore at most
6.4 seconds.

### 7.2 Standard GRU versus minimal GRU

A standard GRU uses the previous hidden state inside several gate and candidate
projections. That makes each step's coefficients depend recursively on the
previous state and makes parallel sequence evaluation difficult.

The MinGRU used here obtains all of its candidate and gate quantities from the
current input:

$$
\begin{bmatrix}
c_t\\
g_t\\
p_t
\end{bmatrix}
=
W_gx_t,
\qquad
W_g\in\mathbb R^{384\times128}.
$$

Split that 384-vector into three 128-vectors.

The positive candidate activation is

$$
\widetilde m_t
=
\phi(c_t),
$$

where elementwise

$$
\phi(c)
=
\begin{cases}
c+0.5,&c\geq0,\\
\operatorname{sigmoid}(c),&c<0.
\end{cases}
$$

This function is continuous and keeps the recurrent candidate positive, which
allows a stable log-domain scan.

The update gate is

$$
z_t=\operatorname{sigmoid}(g_t).
$$

The recurrent state update is

$$
m_t
=(1-z_t)\odot m_{t-1}
+z_t\odot\widetilde m_t.
$$

Interpret one coordinate:

- if \(z_t\approx0\), retain the old memory;
- if \(z_t\approx1\), replace it with the current candidate;
- intermediate values blend old and new information.

The implementation then adds a learned highway gate

$$
q_t=\operatorname{sigmoid}(p_t)
$$

and emits

$$
h_t
=
q_t\odot m_t
+(1-q_t)\odot x_t.
$$

The recurrent state carried to the next time is \(m_t\); the feature sent to
the decoder is \(h_t\). The highway path allows a coordinate to pass the
current encoded observation directly instead of forcing all information
through memory.

### 7.3 Why MinGRU is scan-friendly

Rewrite the recurrent update coordinatewise as

$$
m_t=a_tm_{t-1}+b_t,
$$

where

$$
a_t=1-z_t,
\qquad
b_t=z_t\widetilde m_t.
$$

The pair \((a_t,b_t)\) describes an affine function of the previous state.
Two consecutive affine updates compose as

$$
(a_2,b_2)\circ(a_1,b_1)
=
(a_2a_1,\ b_2+a_2b_1).
$$

This composition is associative. In principle, an associative prefix scan can
evaluate a sequence with much more parallelism than an ordinary GRU whose
gates themselves depend on \(m_{t-1}\).

The current CUDA implementation uses this scan form and log-domain
coefficients for numerical stability. Its fused kernel processes the complete
sequence for each batch/hidden coordinate, looping across the 128 time steps
inside that kernel while parallelizing over batch elements and hidden
coordinates. Thus it removes per-timestep model/kernel orchestration and uses
the scan-friendly recurrence, but the present kernel should not be imagined as
128 wholly independent time steps: the recurrence is still fully respected.

### 7.4 Rollout forward versus training forward

During environment interaction, PufferLib receives one observation at a time.
It uses the stepwise kernel:

$$
(x_t,m_{t-1})\longrightarrow(h_t,m_t).
$$

During training, a minibatch contains complete 128-step sequences. PufferLib
uses `mingru_scan_forward` over each sequence and saves sparse checkpoints
every four steps. The backward scan recomputes intermediate quantities between
those checkpoints to reduce memory traffic.

The backward pass propagates gradients from later losses through earlier
recurrent states. This is backpropagation through time, even though it is
implemented as a custom scan rather than a Python loop.

### 7.5 Decoder

The decoder is

$$
\begin{bmatrix}
\mu_{t,0}\\
\mu_{t,1}\\
\widehat V_t
\end{bmatrix}
=
W_dh_t,
\qquad
W_d\in\mathbb R^{3\times128}.
$$

The first two outputs are Gaussian means. The last is the critic's scalar value
estimate. The log standard deviations are a separate two-element parameter
vector shared by every state.

This means exploration can shrink or grow separately for main and attitude
actions, but it cannot be state-dependent. For example, the policy cannot
learn a high standard deviation at altitude and a low one near touchdown
without expressing that behavior indirectly through the means and clipping.

## 8. How PufferLib samples experience

“Sampling” refers to three different random operations here:

1. the C environment samples a random reset state;
2. the policy samples raw actions from two Gaussians; and
3. PPO samples whole agent sequences from the collected rollout for each
   minibatch.

These should not be conflated.

### 8.1 Number of environments

Booster Landing has one agent per environment. Consequently,
`vec.total_agents` is also the actual number of parallel C environments.

The checked-in inherited default is:

```text
total_agents = 4096
num_buffers = 2
num_threads = 16
```

Most recent serious experiments overrode `total_agents` to 8192. With 8192:

- 8192 independent booster environments run at once;
- each of two vector buffers owns 4096 environments;
- the 16 CPU worker threads are divided into eight OpenMP workers per buffer;
- each buffer has its own GPU stream; and
- the two buffers pipeline GPU policy inference with CPU simulation work.

The `env.num_envs = 1024` entry in the Booster config is not read by the native
binding and does not set the actual count. `vec.total_agents` does.

### 8.2 One rollout step

For each buffer and time index \(t\), the native worker does this:

1. Copy the environment's current observations, previous rewards, and previous
   terminal flags into rollout slot \(t\).
2. Run encoder, recurrent step, and decoder on the GPU.
3. Sample two Gaussian actions on the GPU.
4. Store raw action, old log probability, and value prediction in slot \(t\).
5. Copy actions from GPU to CPU.
6. Call `c_step` for all environments in parallel.
7. Copy the new observations, rewards, and terminal flags to GPU, ready for
   slot \(t+1\).

The rollout tensors initially have time-major layouts:

```text
observations  [T, B, 8]
actions       [T, B, 2]
values        [T, B]
logprobs      [T, B]
rewards       [T, B]
terminals     [T, B]
```

Here \(T=128\), and \(B\) is 4096 by default or 8192 in the recent runs.

At 8192 agents, one rollout contains

$$
128\times8192=1{,}048{,}576
$$

counted agent steps. At the checked-in 4096-agent default, it contains 524,288
steps.

### 8.3 The shifted reward convention

At slot \(t\), the buffer stores:

- current observation \(o_t\);
- action \(a_t\), its log probability, and \(V_t\); but
- the reward and terminal flag produced by the **previous** action.

A small timeline makes this clearer:

| Buffer slot | Observation | Sampled action | Stored reward |
|---:|---|---|---|
| 0 | \(o_0\) | \(a_0\) | reset/previous reward, normally 0 |
| 1 | \(o_1\) | \(a_1\) | reward caused by \(a_0\) |
| 2 | \(o_2\) | \(a_2\) | reward caused by \(a_1\) |

PufferLib's advantage kernel accounts for this by using reward and done at
\(t+1\) when forming the target for action \(t\).

The final action at slot 127 has no slot 128 in which to store its resulting
reward. The advantage kernel therefore computes targets only for slots
0–126; the pre-normalization advantage at slot 127 remains zero. After
advantage normalization, even that zero entry can receive the batch's
mean-offset. A terminal event caused exactly by the last rollout action is
carried into slot 0 of the next rollout and is not normally assigned back to
that preceding action. This is a native boundary edge case worth keeping in
mind.

### 8.4 Recurrent reset and completed-episode padding

With `reset_state = True`, PufferLib zeros every rollout MinGRU state at the
start of each 128-step rollout. It does not selectively zero a native hidden
state whenever an individual environment emits `done`.

The Booster environment works around that mismatch:

- If an episode finishes before the rollout boundary, it emits the terminal
  reward/flag once.
- It then holds the terminal physical observation and returns zero reward
  while waiting.
- At the shared rollout boundary, it resets the physical episode.
- The next episode therefore begins just after PufferLib has zeroed the
  MinGRU state.

This prevents a new episode from inheriting the previous episode's hidden
memory. It also creates padding steps that count toward `global_step` but are
not real flight transitions. The native trainer has no separate padding mask:
after the one terminal flag, those ignored actions and zero-reward held states
still occupy PPO slots and receive bootstrapped targets. The terminal flag
correctly stops the preceding real trajectory, but the padding itself is
treated as a short zero-reward pseudo-sequence.

If an episode is still in progress at the boundary, its physics continues into
the next rollout while its recurrent state is reset. Long flights are therefore
processed as independent 128-step recurrent chunks. `env.rollout_horizon` must
remain equal to `train.horizon`; the current config sets both to 128.

## 9. From a rollout to learning targets

After collection, PufferLib transposes time-major rollout data from
`[T,B,...]` to sequence-major `[B,T,...]`. Each row is one environment's
128-step sequence.

### 9.1 Temporal-difference error

For ordinary on-policy GAE, the one-step temporal-difference residual is

$$
\delta_t
=
r_{t+1}
+\gamma(1-d_{t+1})V_{t+1}
-V_t.
$$

Interpretation:

- \(r_{t+1}+\gamma V_{t+1}\) is a one-step bootstrapped target;
- \(V_t\) is what the critic predicted before seeing the transition;
- their difference is the critic's surprise.

If the next state is terminal, \(d_{t+1}=1\), so the future value is removed.

### 9.2 Generalized advantage estimation

GAE accumulates recent TD residuals:

$$
\widehat A_t
=
\delta_t
+\gamma\lambda(1-d_{t+1})\widehat A_{t+1}.
$$

Expanded:

$$
\widehat A_t
=
\delta_t
+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots.
$$

The current values are

$$
\gamma\lambda
=
0.9999(0.97)
=
0.969903.
$$

Consequently:

- a TD residual 127 steps away has direct weight around 0.021;
- a terminal residual 500 steps away has direct weight around
  \(2\times10^{-7}\).

The critic's bootstrap can still propagate information across many training
iterations, so this is not “zero credit after 128 steps.” It does explain why
dense, well-aligned shaping and a good value function matter for flights that
last roughly 500–600 steps.

The return target for the critic is

$$
\widehat R_t=\widehat A_t+V_t.
$$

### 9.3 PufferLib's GAE/V-trace variation

At the start of training on a fresh rollout, every importance ratio is set to
one, so the first advantage computation is ordinary GAE.

After each PPO minibatch, PufferLib writes the newly computed policy ratios and
new values back into the corresponding rollout rows. Before the next
minibatch, it recomputes advantages for the full rollout using clipped
importance terms:

$$
\bar\rho_t=\min(\rho_t,1),
\qquad
\bar c_t=\min(\rho_t,1).
$$

For the active BF16 build and 128-step horizon, the vectorized kernel uses

$$
\delta_t
=
\bar\rho_t
\left[
r_{t+1}
+\gamma(1-d_{t+1})V_{t+1}
-V_t
\right],
$$

$$
\widehat A_t
=
\delta_t
+\gamma\lambda\bar c_t
(1-d_{t+1})\widehat A_{t+1}.
$$

This is not textbook “freeze all advantages and values for several PPO
epochs.” It is PufferLib's hybrid of PPO, recomputed advantages, and V-trace
style correction.

## 10. How training sequences are sampled

The minibatch size is 8192 **time points**, and the horizon is 128. PufferLib
therefore selects

$$
\frac{8192}{128}=64
$$

complete environment sequences for each optimizer step.

At 8192 agents, the full rollout has 8192 rows. With replay ratio 1:

$$
\text{optimizer steps per rollout}
=
\frac{8192\times128}{8192}
=
128.
$$

Before each one of those 128 optimizer steps, PufferLib:

1. recomputes advantages;
2. computes one priority per 128-step row;
3. samples 64 rows **with replacement**; and
4. trains on the selected \(64\times128=8192\) points.

The priority formula is approximately

$$
p_i
\propto
\left(
\sum_t|\widehat A_{i,t}|
\right)^\alpha.
$$

Current Booster training sets

$$
\alpha=\texttt{prio\_alpha}=0,
$$

so every trajectory has equal probability and the associated importance
weight is effectively one. Sampling remains multinomial with replacement:
some rows can appear repeatedly while others are never selected in that
rollout's 128 updates.

Despite the config name `replay_ratio`, this does not use a long-lived replay
buffer of old experience. It repeatedly samples the just-collected rollout.
The method remains close to on-policy because a new rollout is collected after
these updates.

## 11. PPO for the continuous policy

### 11.1 Gaussian log probability

For one action dimension,

$$
\log\pi(a\mid\mu,\sigma)
=
-\frac12\left(\frac{a-\mu}{\sigma}\right)^2
-\log\sigma
-\frac12\log(2\pi).
$$

The two-dimensional joint log probability is

$$
\log\pi(a_t\mid I_t)
=
\sum_{i=0}^{1}
\log\mathcal N(a_{t,i};\mu_{t,i},\sigma_i^2).
$$

PufferLib stores this value at collection time as
\(\log\pi_\text{old}(a_t\mid I_t)\). During training, the updated network
computes \(\log\pi_\theta\) and the probability ratio

$$
\rho_t(\theta)
=
\exp\left[
\log\pi_\theta(a_t\mid I_t)
-\log\pi_\text{old}(a_t\mid I_t)
\right].
$$

If the policy has not changed, \(\rho=1\).

### 11.2 Advantage normalization

Within each 8192-point minibatch, PufferLib standardizes advantages:

$$
\widetilde A_t
=
\frac{\widehat A_t-\mu_A}{\sqrt{\operatorname{Var}(A)}+10^{-8}}.
$$

This stabilizes gradient scale, but it means “positive” and “negative” in the
PPO loss are relative to the selected minibatch mean.

### 11.3 Clipped policy objective

The unclipped policy-gradient loss would be

$$
L^\text{unclipped}_\pi
=
-\widetilde A_t\rho_t.
$$

PPO also forms

$$
L^\text{clipped}_\pi
=
-\widetilde A_t
\operatorname{clip}(\rho_t,0.8,1.2).
$$

The implemented per-step loss is

$$
L_\pi
=
\max
\left(
L^\text{unclipped}_\pi,
L^\text{clipped}_\pi
\right).
$$

Why the maximum? Training minimizes loss. Taking the worse of the clipped and
unclipped objectives prevents the optimizer from benefiting by moving an
action's probability too far in the already-helpful direction.

For a positive advantage, PPO tries to raise the sampled action's probability,
but stops receiving policy gradient after the ratio grows too far above one.
For a negative advantage, it tries to lower probability, but stops after the
ratio falls too far below one.

### 11.4 Value loss

Let \(V_\text{old}\) be the rollout value, \(V_\theta\) the current prediction,
and \(\widehat R\) the GAE return target. The clipped value candidate is

$$
V_\text{clip}
=
V_\text{old}
+\operatorname{clip}
(V_\theta-V_\text{old},-0.2,0.2).
$$

The value loss is

$$
L_V
=
\frac12
\max
\left[
(V_\theta-\widehat R)^2,
(V_\text{clip}-\widehat R)^2
\right].
$$

This discourages the critic from changing too aggressively on one rollout.

### 11.5 Entropy

The differential entropy of one Gaussian is

$$
\mathcal H_i
=
\frac12\left(1+\log(2\pi)\right)+\log\sigma_i.
$$

The policy entropy is the sum over the two action dimensions. Continuous
differential entropy can legitimately be negative when the standard
deviations become small.

The total loss is

$$
L
=
\mathbb E[L_\pi]
+c_V\mathbb E[L_V]
-c_H\mathbb E[\mathcal H],
$$

with current checked-in

$$
c_V=0.5,
\qquad
c_H=0.
$$

The recent entropy-recovery experiments overrode \(c_H\) to \(10^{-4}\).

## 12. Why policy gradients work without differentiating the rocket

The key identity is

$$
\nabla_\theta J(\theta)
\approx
\mathbb E
\left[
\nabla_\theta\log\pi_\theta(a_t\mid I_t)
\widehat A_t
\right].
$$

The environment only needs to tell us whether the sampled action led to a
better or worse outcome than expected. We differentiate the action's **log
probability**, which is a differentiable neural-network quantity. We never
differentiate position with respect to thrust through `env.c`.

For a Gaussian:

$$
\frac{\partial\log\pi}{\partial\mu}
=
\frac{a-\mu}{\sigma^2},
$$

$$
\frac{\partial\log\pi}{\partial\ell}
=
\frac{(a-\mu)^2}{\sigma^2}-1,
\qquad \ell=\log\sigma.
$$

Suppose the policy sampled unusually high raw throttle and the resulting
advantage was positive. Ignoring PPO clipping, gradient descent shifts the mean
toward that sampled raw action. If the advantage was negative, it shifts the
mean away. The log-standard-deviation gradient also learns whether deviations
of that size should become more or less common.

This is statistical credit assignment, not a label saying “the correct
throttle was 0.73.” Across many randomized flights, the policy distribution
gradually puts more mass on action sequences associated with high return.

The actuator clamp complicates this interpretation. Raw actions 1.2 and 4.0
both produce full physical command, but PPO treats them as distinct Gaussian
samples. The algorithm learns in raw-action space while outcomes are generated
in clipped-action space.

## 13. Exact gradient and weight-update flow

One optimizer step follows this path.

### 13.1 Forward pass

For 64 selected sequences:

1. The initial MinGRU training state is zeroed.
2. All \(64\times128\) observations pass through the encoder.
3. The MinGRU scan builds temporally dependent hidden features.
4. The decoder produces two means and one value at every time step.
5. The global log-standard-deviation vector supplies the two standard
   deviations.
6. The fused PPO kernel recomputes log probabilities, ratios, entropy, policy
   loss, and value loss.

### 13.2 Local output gradients

The fused CUDA PPO kernel analytically computes gradients for:

- each action mean;
- each action log standard deviation; and
- each value prediction.

PPO clipping can set a policy gradient to zero at a clipped sample. That does
not automatically zero the value or entropy gradients at the same sample.

### 13.3 Decoder backward

The decoder combines policy and value derivatives into a three-column output
gradient. Matrix multiplication produces:

- the decoder weight gradient;
- a gradient with respect to each hidden feature; and
- a summed two-element gradient for global `logstd`.

The policy and critic have separate final rows, but their hidden-feature
gradients now meet.

### 13.4 MinGRU backward through time

The MinGRU backward scan sends gradients:

- through the highway gate;
- through the candidate and update gates;
- backward through recurrent state dependencies across the 128 steps; and
- into the MinGRU projection matrix.

A loss near the end of the selected sequence can therefore change how an
earlier observation is encoded into recurrent memory.

### 13.5 Encoder backward

The remaining hidden gradient flows into the encoder, producing a gradient for
the \(128\times8\) observation projection. Because actor and critic share this
path, that gradient contains both policy-control and value-prediction effects.

### 13.6 Global gradient clipping

All parameter gradients are held in one contiguous buffer. PufferLib computes
the global norm and scales the entire gradient if necessary:

$$
g
\leftarrow
g\min\left(
\frac{1.5}{\|g\|+10^{-6}},1
\right).
$$

This preserves the direction of the complete gradient while limiting its
magnitude.

### 13.7 Muon optimizer

The native backend currently uses Muon, not Adam.

First it applies Nesterov momentum with \(\beta_1=0.95\):

$$
m\leftarrow0.95m+g,
$$

$$
u\leftarrow g+0.95m.
$$

For every parameter registered with at least two dimensions, the optimizer
normalizes and approximately orthogonalizes the update using five
Newton–Schulz iterations. It then scales the result according to matrix aspect
ratio. This includes the encoder, decoder, and MinGRU matrices. It also
includes `logstd`: although logically a two-element vector, it is registered
with shape `[1, 2]`, so the current Muon code treats it as a \(1\times2\)
matrix rather than applying a plain scalar/vector momentum update.

The FP32 master weights are updated by

$$
\theta
\leftarrow
\theta-\eta u_\text{Muon}.
$$

Weight decay is currently passed as zero. If learning-rate annealing is
enabled, \(\eta\) follows a cosine schedule from the configured initial rate to
zero across the configured training budget. After every optimizer step, FP32
master weights are cast back to BF16 for the next native forward pass.

`beta2` remains in the generic config but is not used by this Muon
implementation. The configured optimizer `eps` is stored, but the current step
uses fixed numerical constants in its clipping and normalization kernels.
These fields should not be interpreted as Adam hyperparameters for this run.

## 14. A concrete tensor walkthrough

Use the recent 8192-agent experiment settings:

```text
agents              B = 8192
horizon              T = 128
observation size     O = 8
action dimensions    A = 2
hidden size          H = 128
minibatch points     8192
minibatch sequences  8192 / 128 = 64
```

### Collection

```text
observations  [128, 8192, 8]
actions       [128, 8192, 2]
values        [128, 8192]
old logprobs  [128, 8192]
rewards       [128, 8192]
dones         [128, 8192]
```

The policy recurrent state during one rollout buffer is:

```text
[1 layer, 4096 agents in that buffer, 128 hidden units]
```

### Training layout

After transpose:

```text
observations  [8192, 128, 8]
actions       [8192, 128, 2]
...
```

One minibatch selects 64 rows:

```text
mb_obs        [64, 128, 8]
mb_actions    [64, 128, 2]
mb_state      [1, 64, 128]   # zeroed
```

The encoder flattens the first two axes for matrix multiplication, then the
MinGRU views the result as sequences:

```text
encoded       [64, 128, 128]
MinGRU out    [64, 128, 128]
decoder out   [64, 128, 3]
```

The three decoder outputs are:

```text
[..., 0] main-action mean
[..., 1] attitude-action mean
[..., 2] value
```

PPO evaluates 8192 time points, backpropagates once, and Muon takes one step.
That process repeats 128 times before PufferLib collects a new rollout.

## 15. Configuration: three meanings of “current”

There are three different states that should not be collapsed into one.

### 15.1 Current checked-in defaults

The current config file specifies:

| Area | Value |
|---|---:|
| Reset altitude | 100–500 m |
| Reset downward speed | 2–15 m/s |
| Reset horizontal velocity | 0 |
| Reset angular velocity | 0 |
| Touchdown limits | 8 m/s, 2 m/s, 0.06 rad, 0.10 rad/s |
| Hidden size / layers | 128 / 1 |
| Agents | 4096 inherited from `default.ini` |
| Horizon | 128 |
| Minibatch size | 8192 |
| Learning rate | \(3\times10^{-5}\), cosine to zero |
| \(\gamma\) / \(\lambda\) | 0.9999 / 0.97 |
| PPO clip / value clip | 0.2 / 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0 |
| Priority alpha | 0 |
| Replay ratio | 1 |
| Gradient norm | 1.5 |
| Training budget | 1 billion counted agent steps |

This is an easy curriculum configuration, not the full mission.

### 15.2 Best retained continuous checkpoint

The short path

```text
checkpoints/stable_1000.bin
```

is a symlink to the 105,906,176-step checkpoint from run
`1784154576879`. It contains 202,248 bytes of FP32 checkpoint slots and has
SHA-256:

```text
c430a77b22ebb8194ab9b3b5dd90c123feb84ec496d27ad528ec2a069a6628fe
```

It was selected by fixed independent evaluation, not because it was the final
checkpoint of its run.

### 15.3 Latest completed local experiment

The most recent log by modification time is run `1785064592220`. It:

- loaded `checkpoints/stable_1000.bin`;
- trained on the easy 100–1000 m distribution;
- used 8192 agents;
- used \(10^{-5}\) initial learning rate;
- used entropy coefficient \(10^{-4}\);
- used the then-current expanded potential with stopping and fuel risks added
  independently; and
- ran for 1.5 billion counted steps.

This experiment reproduced the earlier
`100_1000_independent_fuel_risk` setup. It is the latest run, but not the best
policy.

### 15.4 Loading is weights-only

`load_model_path` loads the flat neural-network weights. It does not restore:

- Muon momentum;
- optimizer step count;
- learning-rate schedule position;
- rollout state; or
- RNG state.

A continuation therefore begins with zero optimizer momentum and restarts its
cosine schedule from the newly configured initial learning rate. It is not an
exact resumption of the source run.

## 16. Current empirical state

### 16.1 Physics and action mapping are feasible

The deterministic continuous controller establishes that the simulator is
controllable:

| Evaluation | Controller result |
|---|---:|
| Fixed hard 8192-episode suite | 8183/8192 = 99.8901% |
| Three independent seeds | 24,544/24,576 = 99.8698% |
| Exact 1500 m | 8180/8192 = 99.8535% |

On the broad fixed suite it retained about 34.84 kg of fuel, used about
63.14 kg of main fuel and 2.02 kg of side fuel, and its only failures were
horizontal touchdown-speed misses. This is strong evidence that fuel capacity
and actuator authority are not the fundamental bottlenecks.

### 16.2 Best retained RL result

On the easy 100–1000 m distribution, `stable_1000.bin` scored:

```text
8070 / 8192 = 98.5107%
```

and a second independent 8192-episode evaluation scored 98.5718%.

This is currently the best validated continuous checkpoint for that easy
100–1000 m curriculum. It does not establish success on the strict, randomized
100–1500 m mission.

### 16.3 Latest independent-fuel-risk run regressed

The latest run's first downsampled score, after about 110 million new steps,
was 88.66%. Its final evaluation was:

| Metric | Final value |
|---|---:|
| Overall score | 82.9147% |
| 100–500 m score | 99.6967% |
| 500–750 m score | 99.9012% |
| 750–1000 m score | 38.6991% |
| Mean terminal fuel | 3.23 kg |
| Mean main fuel used | 92.92 kg |
| Fuel-empty failure | 11.87% |
| Vertical-speed failure | 7.16% |
| Horizontal-speed failure | 6.38% |
| Off-pad failure | 0% |

The per-bucket split is more informative than the aggregate. The policy became
excellent below 750 m while sacrificing the upper quarter of the curriculum
and consuming almost all fuel. Separating fuel risk in the shaping potential
did not solve the high-altitude behavior.

### 16.4 What remains unsolved

The continuous RL line has not yet demonstrated a stable solution for all of:

- altitude 100–1500 m;
- downward speed 5–25 m/s;
- horizontal velocity \(\pm5\) m/s;
- initial angular velocity \(\pm0.05\) rad/s; and
- strict touchdown limits of 5 m/s vertical, 1 m/s horizontal, 0.03 rad
  angle, and 0.05 rad/s angular velocity.

The dominant research problem is no longer basic physical feasibility. It is
stable long-horizon learning and retention across altitude buckets. Policies
often reach a strong checkpoint, then drift or collapse while aggregate PPO
diagnostics remain numerically ordinary.

## 17. Current implementation caveats and open questions

### 17.1 Environment reward versus training reward

The environment-side potential is gamma-matched, but the final combined reward
is clipped to \([-1,1]\). The logged return and critic target now agree;
textbook potential invariance is still not exact on transitions where the
combined reward reaches the clamp.

### 17.2 Raw Gaussian action and squashed actuator

The environment now applies an invertible `tanh` transform to each raw
Gaussian sample. PPO continues to use raw actions and raw log probabilities;
the fixed transform's Jacobian cancels from old/new likelihood ratios. Because
the transform changes physical action semantics, checkpoints trained with the
older clipping map are shape-compatible but are not guaranteed to preserve
behavior. The retained `stable_500.bin` was checked explicitly under the new
map and still solves its complete `100-500 m` source range.

### 17.3 Rollout boundary target

The shifted buffer convention correctly assigns rewards for actions 0–126 but
has no in-rollout successor slot for action 127. A terminal transition exactly
there can lose its natural action assignment.

### 17.4 Recurrent state is chunked, not episode-long

Hidden state resets every 128 steps. Completed episodes are padded until a
boundary so new episodes start cleanly, while ongoing flights lose recurrent
history at each boundary. This behavior should be understood before changing
the horizon.

### 17.5 Counted steps include padding

An environment that lands early can spend the rest of its rollout idle.
PufferLib still increments global agent steps for those slots. A billion
reported steps is therefore not necessarily a billion physical control
transitions.

### 17.6 Default config is not the target or best run

Running the config without overrides gives 4096 agents and a 100–500 m easy
curriculum. It neither reproduces the 8192-agent latest experiment nor tests
the full strict benchmark.

### 17.7 Continuous native cleanup fault

The ordinary native continuous training path saves its final checkpoint and
then can segfault during `_C.close`. The same issue was reproduced with
`squared_continuous`, so it is not specific to this environment's layout.
`scripts/train_booster_continuous.py` defers explicit native cleanup, lets the
normal JSON log be written, and exits so the OS reclaims resources.

### 17.8 Working-tree status

At this snapshot, the continuous environment/config/benchmark line is still
working-tree material rather than a clean committed baseline, and relevant
trainer/evaluator files also have local modifications. “Current behavior”
therefore means this exact workspace, not necessarily repository `HEAD`.

### 17.9 Flat parameter alignment deserves a fix

A source-level audit found a packing mismatch in the generic native allocator:

- each registered tensor begins at a 16-byte-aligned address;
- `Allocator.total_elems` counts only logical elements and excludes alignment
  padding; but
- the flat parameter, gradient, master-weight, optimizer, cast, and checkpoint
  views use that unpadded element count over the raw allocation.

For this BF16 model, the encoder and decoder matrices happen to end on aligned
boundaries. The `[1, 2]` `logstd` consumes four bytes, after which six BF16
padding elements are inserted before the MinGRU matrix. The flat view includes
those six padding positions and stops six elements before the physical end of
the MinGRU allocation.

This means the nominal count and 202,248-byte checkpoint size are internally
consistent, but the current flat view is shifted relative to the MinGRU
tensor: padding is serialized and optimized as if it were parameter data,
while the final six physical MinGRU elements are outside that flat view.
Reload remains repeatable under the fixed initialization seed, which is why
checkpoint tests can still reproduce behavior.

This is a confirmed layout mismatch in the source, but it has not been
isolated experimentally as the cause of policy collapse. That causal claim
should not be made without fixing the packing and running a controlled
comparison.

## 18. How to read the metrics

### `env/score`

The fraction of completed episodes that satisfy every touchdown criterion.
This is the primary objective metric.

### `env/episode_return`

The sum of the environment-clamped shaping and terminal rewards. It is the same
reward stream used by the critic, but remains a diagnostic rather than a
substitute for score.

### `loss/value`

How well the critic fits current bootstrapped return targets. A small value loss
does not prove the policy is good; a critic can accurately predict poor
outcomes.

### `loss/kl` and `loss/clipfrac`

These measure how far policy updates move from the rollout policy and how often
PPO clipping activates. Ordinary-looking values do not rule out slow policy
drift across hundreds of millions of steps.

### `loss/entropy`

The differential entropy of the two Gaussians. It may be negative. Falling
entropy means standard deviations are shrinking and the policy is becoming
more deterministic.

### Continuous-action diagnostics

`main_throttle_near_off_rate`, `main_throttle_partial_rate`, and
`main_throttle_near_full_rate` partition physical throttle into `0-5%`,
`5-95%`, and `95-100%`. `attitude_command_near_limit_rate` reports the
fraction of commands with at least 95% signed attitude authority.

`mean_raw_main_action`, `mean_absolute_raw_main_action`, and
`mean_absolute_raw_attitude_action` expose raw policy-output magnitude. The
altitude-phase metrics report post-squash throttle and absolute attitude
commands separately for `0-100 m`, `100-500 m`, and `500+ m`. These metrics
change no actions or rewards; they distinguish policy-distribution drift from
actual actuator behavior.

### Altitude-bucket scores

These are essential. A high aggregate can hide catastrophic forgetting in a
smaller or harder altitude band. The recent run's 82.9% aggregate concealed
99.9% performance in one band and only 38.7% in another.

### Fuel and touchdown metrics

`terminal_fuel`, `main_fuel_used`, touchdown speed, and overlapping failure
reasons reveal *how* a policy changed. They distinguished a physical
limitation from the observed learned behavior of burning more than 92 kg.

## 19. Common misconceptions

### “Is the value network predicting success probability?”

Not exactly. It predicts expected discounted, shaped, trainer-clamped return
under the current policy. It can correlate strongly with success probability,
especially when terminal \(\pm1\) dominates, but those are not mathematically
identical targets.

### “Does the network output the best action?”

It outputs means of two stochastic action distributions. PufferLib samples
from them during collection and evaluation. The learned standard deviations
control exploration.

### “Is observation 7 the value function?”

No. Observation 7 is an analytic safety-risk feature provided by the
environment. The value function is a learned scalar emitted by the decoder
after the MinGRU.

### “Does PPO compare the chosen action with a known correct action?”

No. It compares the trajectory's estimated advantage with the probability of
the action that was actually sampled.

### “Does backpropagation run through the simulator?”

No. Gradients stop at the action-distribution calculation. Rewards from the C
simulator weight log-probability gradients.

### “Does a recurrent network remember the whole landing?”

No. In the current native setup it remembers within a 128-step, 6.4-second
rollout chunk. Its state is zeroed at the next chunk.

### “Does one PPO minibatch contain 8192 independent flights?”

No. It contains 64 complete 128-step sequence rows, for 8192 time points.

### “Does `num_envs = 1024` mean there are 1024 boosters?”

Not in this native binding. Because there is one agent per environment,
`vec.total_agents` determines the count: 4096 by default and 8192 in recent
experiments.

### “Does loading a checkpoint resume training exactly?”

No. It restores weights only. Muon momentum and the cosine-schedule position
restart.

## 20. Source-to-concept map

| Question | Where the answer lives |
|---|---|
| What is the physical state? | `booster_landing_continuous.h`, `env.c` |
| How are observations normalized? | `compute_observations` in `env.c` |
| How are actions squashed/mapped? | `main_throttle_from_action`, `attitude_command_from_action` |
| What are the equations of motion? | `c_step` in `env.c` |
| How is reward shaped? | risk helpers and `landing_safety_potential` in `env.c` |
| What counts as success? | terminal block in `c_step` |
| Why are actions continuous? | `NUM_ATNS` and `ACT_SIZES` in `binding.c` |
| How many environments run? | `vec.total_agents` and `src/vecenv.h` |
| How are Gaussian actions sampled? | `sample_logits` in `src/pufferlib.cu` |
| What is the MinGRU equation? | `mingru_gate` and scan kernels in `src/models.cu` |
| How are advantages computed? | `puff_advantage_*` in `src/pufferlib.cu` |
| How is PPO clipped? | `ppo_loss_compute` in `src/pufferlib.cu` |
| How are weights updated? | `muon_step` in `src/muon.cu` |
| Why does the launcher bypass close? | `scripts/train_booster_continuous.py` |
| What happened in past runs? | `experiments/booster_landing.md` and local JSON logs |

## Appendix: What worked during training

This appendix is intentionally outcome-oriented. “Worked” means a change
produced a reproducible improvement, removed a diagnosed failure mode, or
created a trustworthy way to select policies. It does not mean the final
strict continuous mission is solved.

### A. Foundations established by the discrete lineage

1. **Make recurrent episode boundaries agree with rollout boundaries.**
   Holding completed environments idle until the shared boundary prevented a
   new episode from inheriting stale MinGRU memory. This was necessary for
   coherent recurrent training.

2. **Normalize observations at mission-scale constants.**
   Using 2000 m for altitude and 200 m/s for vertical velocity kept observation
   scales stable while the altitude curriculum expanded. Changing
   normalization every stage had made checkpoint continuation inconsistent.

3. **Expose fuel and a physics-derived viability feature.**
   Fuel fraction and the stopping/fuel safety feature gave the network direct
   access to variables needed for long-horizon braking decisions.

4. **Use gamma-matched potential shaping while keeping terminal success
   explicit.**
   Dense changes in landing viability were much easier to learn from than a
   distant terminal signal alone. Terminal \(+1/-1\) kept score interpretable.
   The remaining native reward clamp is now the important qualification.

5. **Train by curriculum and preserve intermediate checkpoints.**
   Expanding altitude in stages—rather than beginning with the full
   100–1500 m distribution—reliably produced capable policies. The best
   checkpoint was often in the middle of a run, not at its end.

6. **Use small continuation learning rates.**
   The large generic rate \(0.015\) destroyed already capable strict policies.
   Rates around \(3\times10^{-5}\) in later strict work, and carefully reduced
   rates for continuation, preserved behavior much better.

7. **Use uniform sequence sampling for this task.**
   `prio_alpha = 0` was more stable than strongly prioritizing rare,
   high-absolute-advantage trajectories. Priority 0.8 repeatedly overfocused
   rare failures and destabilized near-solved policies.

8. **Canonicalize horizontal symmetry.**
   Reflecting positive-initial-velocity episodes into a fixed policy frame
   removed a major directional bias. A prior fixed \(-5/+5\) m/s score gap of
   about 6.31 percentage points fell to about 0.06 points. A reflected
   fine-tuned discrete policy reached roughly 99.69% over 24,576 broad
   independent episodes.

9. **Evaluate fixed and independent reset seeds.**
   Training windows were highly non-monotonic. Large fixed suites, multiple
   independent seeds, exact-1500 m tests, and separate directional tests
   prevented selection based on a lucky or unrepresentative training window.

10. **Use altitude buckets and failure composition, not aggregate score
    alone.**
    This revealed policies trading low-altitude competence for high-altitude
    competence and exposed fuel exhaustion, lateral speed, or angle as
    distinct failure modes.

11. **Build a deterministic controller baseline.**
    The discrete controller achieved about 99.78% over three seeds and 99.84%
    at exactly 1500 m. That separated “the task is physically impossible” from
    “the learning system has not found or retained the controller.”

### B. What has worked in the continuous line

1. **Two continuous heads are a valid and controllable action interface.**
   Full/coast main guidance plus continuous PD attitude control achieved
   99.8901% on the fixed hard suite, 99.8698% over three seeds, and 99.8535% at
   exactly 1500 m. It eliminated simultaneous opposing side commands and used
   little side fuel.

2. **Native continuous PPO, checkpoint save, and reload all run end to end.**
   The backend correctly recognizes two continuous dimensions. It writes the
   expected 202,248-byte flat file, and selected checkpoints reproduce their
   evaluated behavior after reload. The allocator-packing caveat in Section
   17.9 still needs correction; reproducible reload does not make that layout
   ideal.

3. **The cleanup wrapper makes long native continuous runs operational.**
   Deferring `_C.close` avoids the post-save native cleanup crash long enough
   for the normal JSON experiment log to be serialized.

4. **A genuinely fresh easy curriculum can learn.**
   Fresh run `1783869226572` on 100–250 m reached a logged 99.829% around
   2.360 billion steps and remained above 99.7% at the end. This demonstrated
   that continuous PPO was capable of learning rather than merely executing
   the classical controller.

5. **Staged altitude continuation reached a strong 1000 m policy.**
   After selecting and rolling back intermediate stages rather than trusting
   final checkpoints, the retained `stable_1000.bin` reached 98.5107% and
   98.5718% on two independent easy 100–1000 m evaluations.

6. **A small entropy bonus slowed variance collapse.**
   `ent_coef = 0.0001` kept Gaussian entropy higher than the zero-entropy run.
   It did not solve long-run collapse, but the mechanism behaved as intended
   and bought more exploration time.

7. **The expanded potential improved the diagnosed lateral signal.**
   Adding projected pad position, horizontal speed, projected angle, and
   angular-rate risks supplied dense credit for terminal criteria that were
   previously hundreds of steps away. The first shaped recovery improved
   lateral control, even though it did not beat the retained total score.

8. **Keeping observation 7 compatible while changing shaping was successful
   engineering.**
   The safety observation retained its old `max(stop,fuel)` definition, so the
   existing checkpoint's actions were unchanged after rebuilding. The fixed
   evaluation reproduced exactly 8070/8192 successes.

### C. Negative results that should not be repeated blindly

- Starting continuous PPO directly on the full strict range did not discover a
  useful policy.
- A strong checkpoint can gradually collapse without a KL or value-loss
  explosion; longer training is not automatically better.
- Zero entropy coefficient allowed the Gaussian policy to become extremely
  narrow, but a \(10^{-4}\) entropy bonus only slowed rather than prevented
  collapse.
- Large continuation learning rates destroyed learned behavior. Lower rates
  slowed drift but did not by themselves eliminate it.
- Strong priority sampling over rare failures was harmful near high success.
- Aggregate score hid severe altitude forgetting.
- Reward variants that paid for fuel or near-miss quality could create
  higher-return failure strategies unless success remained lexicographically
  preferable.
- Independently adding fuel viability to that expanded potential did not
  produce a better 100–1000 m policy. The run finished at 82.9147%, used about
  92.92 kg of main fuel, and retained only 38.70% success in the 750–1000 m
  bucket.
- The final checkpoint is frequently worse than an earlier checkpoint.
  Checkpoint screening is part of training, not an optional afterthought.

### D. The most reliable recipe so far

The evidence to date supports this practical recipe:

1. Start from a fixed, easy distribution and unchanged observation
   normalization.
2. Train with terminal success/failure explicit and gamma-matched dense
   viability shaping.
3. Use `prio_alpha = 0`, a conservative learning rate, and full 128-step
   sequences.
4. Save frequently.
5. Evaluate candidate checkpoints on large fixed suites and independent reset
   seeds.
6. Require strong scores in every altitude bucket before expanding the range.
7. Change only one difficulty dimension at a time: altitude, then touchdown
   limits, then horizontal velocity, then angular velocity.
8. Keep the deterministic controller result as a feasibility ceiling and fuel
   sanity check.
9. Roll back immediately when a continuation trades away a previously solved
   bucket.
10. Treat `checkpoints/stable_1000.bin`, not the latest completed run or the
    checked-in easy defaults, as the current best continuous RL starting point.
