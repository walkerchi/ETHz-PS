# Adversarial Uncertainty Quantification in Physics-Informed Neural Networks
<center> Yibo Yang, Paris Peerdikaris</center>

---

## Introduction

<center><font size=20>UQPINN = GAN + PINN</font></center>

- **UQPINN** : <font size=5> Uncertainty Quantification Physics-Informed Neural Network</font>
- **GAN** : <font size=5>Generative Adversarial Network </font>
- **PINN** : <font size=5> Physics-Informed Neural Network </font>

**Novalty** : <font size=4>"we will develop a flexible <font color="orange">variational inference</font> framework that will allow us to train such models directly from <font color="orange">noisy input/output data</font>, and predict outcomes of non-linear dynamical systems that are partially <font color="orange">observed</font> with quantified <font color="orange">uncertainty</font>"</font>

---

## Reproduce

### ODE 

$$
\frac{\partial^2 u}{\partial x^2} - u^2 \frac{\partial u}{\partial x}  =f(x)\quad  x\in[-1,1]
$$
$$
f(x) = -\pi^2 sin(\pi x) - \pi cos(\pi x)sin(\pi x)^2
$$
$$
u(x)  \sim \mathcal N(sin(\pi), noise) x=\{-1,1\}
$$

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/ODE_UQPINN/losses.png)

=== 

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/ODE_UQPINN/x_y_uncertainty.png)

---

### Burgers

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
 \quad x\in[-1,1], t\in[0,1]
$$
$$
u(0,x) = -sin(\pi x)
$$
$$
u(t, x) = 0 \quad  x=\{-1, 1\}
$$
$$
\nu = \frac{0.01}{\pi}
$$

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Burgers_UQPINN/losses.png)

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Burgers_UQPINN/y_distribution_2D.png)

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Burgers_UQPINN/x_y_uncertainty.png)

---

### Darcy

$$
\nabla_{\overrightarrow x} (K(u)\nabla_{\overrightarrow x} u(\vec x)) = 0   \quad{\overrightarrow x}\in [0,L_1]\times [0,L_2]
$$
$$
u(\vec x) = 0 \quad x_1 = L_1
$$
$$
-K(u) \frac{\partial u(\vec x)}{\partial x_1} = q \quad x_1 = 0
$$
$$
K(u)  = K_s \sqrt{s(u)} \left(1-(1-s(u)^{\frac{1}{m}})^m\right)^2 \quad x_2 = \{0, L_2\}
$$
$$
s(u) = \left(1 + \left(\alpha (u_g - u)\right)^{\frac{1}{1-m}}\right)^{-m}
$$

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Darcy_UQPINN/losses.png)

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Darcy_UQPINN/y_distribution_2D.png)

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/Darcy_UQPINN/y_relation_2D.png)



---

## Comparison

### ODE

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_ODE_pinn_uqpinn/losses.png)

=== 

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_ODE_pinn_uqpinn/x_y_relation_2D.png)

---

### Burgers

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Burgers_pinn_uqpinn/losses.png)

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Burgers_pinn_uqpinn/y_distribution_2D.png)

---

### Darcy

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Darcy_pinn_uqpinn/losses.png)

===

![img](https://raw.githubusercontent.com/walkerchi/Physics-Seminar/main/output/compare_Darcy_pinn_uqpinn/y_relation_2D.png)



---

## Future

- fix Darcy
- maybe add more baseline