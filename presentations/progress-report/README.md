## Introduction

<center><font size=18>UQPINN = GAN + PINN</font></center>

- **UQPINN** : Uncertainty Quantification Physics-Informed Neural Network
- **GAN** : Generative Adversarial Network
- **PINN** : Physics-Informed Neural Network

**Novalty** : "we will develop a flexible <font color="orange">variational inference</font> framework that will allow us to train such models directly from <font color="orange">noisy input/output data</font>, and predict outcomes of non-linear dynamical systems that are partially <font color="orange">observed</font> with quantified <font color="orange">uncertainty</font>"

----

for GAN training
$$
\begin{aligned}
& \underset{\psi}{max}~\mathcal L_{\mathcal D}(\psi)
\\
& \underset{\theta, \phi}{min}~\mathcal L_{\mathcal G}(\theta, \phi) + \beta~\mathcal L_{PDE} (\theta)
\end{aligned}
$$

for PDE loss
$$
\mathcal L_{PDE} (\theta) = \frac{1}{N_r}\sum_{i=1}^{N_r}\Vert p_\theta(x_i,t_i,z) -u_i\Vert^2
$$

- $N_r$ number of observation points
- $x_t,t_i,u_i$ : observations

for discirminator
$$
\begin{aligned}
&\underset{\psi}{argmin}\frac{\rho(y=+1|x,t,u)}{\rho(y=-1|x,t,u)}
\\
&p_\theta(x,t,u) = \rho(x,t,u|y=+1)
\\
&q(x,t,u) = \rho(x,t,u|y=-1)

\end{aligned}
$$

therefore 

use $T_\psi$ to approximate $\rho(y=+1|x,t,u)$
$$
\begin{aligned}
\mathcal L_D(\psi) = &\mathbb E_{q(x,t)p(z)}[log ~\sigma(T_\psi(x,t,f_\theta(x,t,z)))] + 
\\
&\mathbb E_{q(x,t,u)}[log(1-\sigma(T_\psi(x,t,u)))]
\end{aligned}
$$

for generator 

suppose there is a latent variable $z$ 
$$
\begin{aligned}
p_\theta(x,t,u,z) = p_\theta(u|x,t,z)p(x,t,z)
\end{aligned}
$$

$$
\underset{\theta}{argmax}~\mathbb{KL}\left[
p_\theta(x,t,u)
\Vert
q(x,t,u)\right]
$$

use $f_\theta(u|x,t,z)$ to approximate $p_\theta(u|x,t,z)$ , $p_\theta(u|x,t,z) = \delta(u-f_\theta(x,t,z))$

use $q_\phi$ to variational approximate $q_\phi(z|x,t,u)$

$\lambda > 1$
$$
\mathcal L_G(\theta,\psi) = \mathbb E_{q(x,t)p(z)}
[T_\psi(x,t,f_\theta(x,t,z))+(1-\lambda )log(q_\phi(z|x,t,f_\theta(x,t,z)))]
$$
for prediction 
$$
\mu_u(x^*,t^*) = \mathbb E_{p_\theta}[u|x^*,t^*,z]\approx \frac{1}{N_s}\sum_{i=1}^{N_s}f_\theta(x^*,t^*,z_i)
\\
\sigma_u^2(x^*, t^*) = \mathbb Var_{p_\theta}[u|x^*,t^*,z] \approx\frac{1}{N_s}\sum_{i=1}^{N_s}[f_\theta(x^*,t^*,z_i)-\mu_u(x^*,t^*)]^2
$$
$z_i\sim p(z)$, but actually $z_i\sim Uniform[0,1]$



---

## Reproduce

### ODE
$$
\begin{aligned}
\frac{\partial^2 u}{\partial x^2} &- u^2 \frac{\partial u}{\partial x}  f(x) & x&\in[-1,1]
\\
f(x) &= -\pi^2 sin(\pi x) - \pi cos(\pi x)sin(\pi x)^2
\\
u(x) & \sim \mathcal N(sin(\pi), noise)& x&=\{-1,1\}
\end{aligned}
$$
---
---

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/ODE_UQPINN/losses.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/ODE_UQPINN/x_y_uncertainty.png)

---

### Burgers
$$
\begin{aligned}
\frac{\partial u}{\partial t} &+ u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
 \quad &x&\in[-1,1], t\in[0,1]
\\
u(0,x) &= -sin(\pi x)
\\
u(t, x) &= 0 \quad & x&=\{-1, 1\}
\\
\nu &= \frac{0.01}{\pi}
\end{aligned}
$$
![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Burgers_UQPINN/losses.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Burgers_UQPINN/y_distribution_2D.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Burgers_UQPINN/x_y_uncertainty.png)

---

### Darcy
$$
\begin{aligned}
\nabla_\vec x &(K(u)\nabla_\vec x u(\vec x)) = 0  & \vec x&\in [0,L_1]\times [0,L_2]
\\
u(\vec x) &= 0 & x_1 &= L_1
\\
-K(u) &\frac{\partial u(\vec x)}{\partial x_1} = q & x_1 &= 0
\\
K(u) & = K_s \sqrt{s(u)} \left(1-(1-s(u)^{\frac{1}{m}})^m\right)^2 & x_2 &= \{0, L_2\}
\\
s(u) &= \left(1 + \left(\alpha (u_g - u)\right)^{\frac{1}{1-m}}\right)^{-m}
\end{aligned}
$$

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Darcy_UQPINN/losses.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Darcy_UQPINN/y_distribution_2D.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Darcy_UQPINN/y_relation_2D.png)



---

## Comparison

### ODE

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_ODE_pinn_uqpinn/losses.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_ODE_pinn_uqpinn/x_y_relation_2D.png)

---

### Burgers

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Burgers_pinn_uqpinn/losses.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Burgers_pinn_uqpinn/y_distribution_2D.png)

---

### Darcy

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Darcy_pinn_uqpinn/losses.png)

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Darcy_pinn_uqpinn/y_relation_2D.png)



---

## Future

- fix Darcy
- maybe add more baseline