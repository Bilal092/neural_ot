Create environment using environment.yml.

Details to run are provided with each experiment.

In order to understand the principle behind paper, please start with static subset alignment between toy point distributions start with this [notebook](static_subsetting_toy.ipynb).
| Static example c=2 | Dynamic example c=2 |
|:--------------:|:---------------:|
| <img src="images/squares_c2_potential.png" style="width:500px; height:250px; object-fit:contain;"/> | <img src="images/squares_c2_potential_t1.png" style="width:500px; height:250px; object-fit:contain;"/> |
Evolution of dynamic potential with time
<img src="images/d_squaresc2_transition.png" style="width:500px; object-fit:contain;"/>
MNIST $\rightarrow$ EMNIST

| Static | Dynamic |
|:--------------:|:---------------:|
| <img src="images/MNIST_EMNIST_static_.png" style="width:500px; height:250px; object-fit:contain;"/> | <img src="images/MNIST_EMNIST_dynamic_ode.png" style="width:500px; height:250px; object-fit:contain;"/> |

## FFHQ: Results

### Old → Young

| (a) Static subset | (b) Dynamic subset (Euler 100 steps) |
|:--:|:--:|
| <img src="images/ADULT_YOUNG_static.png" style="width:600px; height:300px; object-fit:contain;"/> | <img src="images/ADULT_YOUNG_dynamic_ode.png" style="width:600px; height:300px; object-fit:contain;"/> |

<p align="center">
  <sub><b>Figure 1.</b> FFHQ old → young translation using (a) static and (b) dynamic subset selection. The dynamic version is evaluated using Euler integration (100 steps).</sub>
</p>

---

### Young → Old

| (a) Static subset | (b) Dynamic subset (Euler 100 steps) |
|:--:|:--:|
| <img src="images/YOUNG_ADULT_static.png" style="width:500px; height:300px; object-fit:contain;"/> | <img src="images/YOUNG_ADULT_dynamic_ode.png" style="width:500px; height:300px; object-fit:contain;"/> |

<p align="center">
  <sub><b>Figure 2.</b> FFHQ young → old translation using (a) static and (b) dynamic subset selection. The dynamic version is evaluated using Euler integration (100 steps).</sub>
</p>

---

### Woman → Man

| (a) Static subset | (b) Dynamic subset (Euler 100 steps) |
|:--:|:--:|
| <img src="images/WOMAN_MAN_static.png" style="width:500px; height:300px; object-fit:contain;"/> | <img src="images/WOMAN_MAN_dynamic_ode.png" style="width:500px; height:300px; object-fit:contain;"/> |

<p align="center">
  <sub><b>Figure 3.</b> FFHQ woman → man translation using (a) static and (b) dynamic subset selection. The dynamic version is evaluated using Euler integration (100 steps).</sub>
</p>

---

### Man → Woman

| (a) Static subset | (b) Dynamic subset (Euler 100 steps) |
|:--:|:--:|
| <img src="images/MAN_WOMAN_static.png" style="width:500px; height:300px; object-fit:contain;"/> | <img src="images/MAN_WOMAN_dynamic_ode.png" style="width:500px; height:300px; object-fit:contain;"/> |

<p align="center">
  <sub><b>Figure 4.</b> FFHQ man → woman translation using (a) static and (b) dynamic subset selection. The dynamic version is evaluated using Euler integration (100 steps).</sub>
</p>

