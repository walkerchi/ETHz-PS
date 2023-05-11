import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline

import tikzplotlib

def rotate_transform(nodes, origin=None, degree=0):
    if origin is None:
        x, y = 0, 0 
    else:
        x, y = origin
    rad = np.deg2rad(degree)
    cos = np.cos(rad)
    sin = np.sin(rad)
    print(cos)
    T = np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]  
        ])
    T_ = np.array([
        [1, 0, -x],
        [0, 1, -y],
        [0, 0, 1]
    ])
    R  = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])
    T = T @ R @ T_


    nodes_pad = np.hstack((nodes, np.ones((nodes.shape[0], 1))))
    nodes_pad = nodes_pad @ T.T
    assert (nodes_pad[:, -1] == 1).all(), f'The last column of nodes_pad should be 1 {nodes_pad}'
    return nodes_pad[:, :-1]


class Trapezoid:
    def __init__(self, x, y, 
                 rotation=0, 
                 top=1,
                 bottom=2, 
                 hight=1,
                 **kwargs):
        a = top / 2 
        b = bottom / 2
        self.center = np.array([x, y])
        self.kwargs = kwargs
        self.nodes = np.array([
            [x-b, y],
            [x+b, y],
            [x+a, y+hight],
            [x-a, y+hight]
        ]) 
        self.rotate(-rotation)
    def rotate(self, degree):
        self.nodes = rotate_transform(self.nodes, self.center, degree)
    @property
    def top(self):
        return self.nodes[:2].mean(0)
    @property
    def bottom(self):
        return self.nodes[2:].mean(0)
    def draw(self, ax):
        ax.add_patch(patches.Polygon(xy=self.nodes, **self.kwargs))

class Arrow:
    def __init__(self, from_, to_, color, width=1):
        self.color = color
        self.width = width
        self.from_ = from_
        self.to_ = to_

        p1 = self.from_.copy()
        p1[0] += 0.1
        p2 = self.to_.copy()
        p2[0] -= 0.1
        self.points = np.stack([
            self.from_,
            p1, 
            self.to_,
            p2
        ])

    def draw(self, ax):
        m = make_interp_spline(self.points[:, 0], self.points[:, 1])
        xs = np.linspace(self.points[:, 0].min(), self.points[:, 0].max(), 10)
        ys = m(xs)
        ax.plot(xs, ys, color=self.color, linewidth=self.width)




def draw_gan():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    # ax.grid()

    generator = Trapezoid(4, 4, top=2, bottom=4, hight=2, rotation=-90, color="#C8BFE7")
    discriminator = Trapezoid(7, 7, top=2, bottom=4, hight=2, rotation=90, color="#EFE4B0")

    ax.text(3, 2, "Generator", fontsize=14, ha="center", va="center")
    ax.text(8, 5, "Discriminator", fontsize=14, ha="center", va="center")
    ax.text(0, 4, "$z\sim~p(z)$", fontsize=20)
    ax.text(5.5, 4, "$p_\\theta(x)$", fontsize=20)
    ax.text(5.5, 8, "$q(x)$", fontsize=20)
    ax.text(9.5, 7, "$\\rho$", fontsize=20)
    ax.text(3, 4, "$f_\\theta$", fontsize=24, va="center", ha="center")
    ax.text(8, 7, "$T$", fontsize=24, va="center", ha="center")

    plt.arrow(1.6, 4, 0.3, 0, color="#3282F6", width=0.02) # z -> f
    plt.arrow(4, 4, 1.4, 0, color="#3282F6", width=0.02)# f -> p
    plt.arrow(6.3, 4.5, 0.65, 1.5, color="#3282F6", width=0.02) # p->t
    plt.arrow(6.3, 8, 0.65, -0.5, color="#3282F6", width=0.02) # q -> t
    plt.arrow(9, 7, 0.4, 0, color="#3282F6", width=0.02) # t-> rho
    generator.draw(ax)
    discriminator.draw(ax)

    plt.show()
    # tikzplotlib.save("gan.tex")

if __name__ == '__main__':
    draw_gan()