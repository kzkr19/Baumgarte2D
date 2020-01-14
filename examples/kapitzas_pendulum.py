import baumgarte2d
from sympy import symbols, sin
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_FILE_PATH: str = "./examples/temp"


def main():
    a = 0.1
    l = 1.0
    g = 9.8
    c = 4.0
    # 振動の角速度がsqrt(2*g*l)/aより大きければ安定する
    # https://en.wikipedia.org/wiki/Kapitza%27s_pendulum
    nu = 2*np.sqrt(2*g*l)/a
    t = symbols("t")

    # 剛体の定義
    block = baumgarte2d.RigidBody(0)

    # 描画の設定
    block.width = 2*l
    block.height = block.width / 10.0

    # 重力の追加
    block.add_force_y(0-block.m*g)

    # 初期値の設定．順にx0, y0, theta0
    block.initial_position = np.array([1/np.sqrt(2), 1/np.sqrt(2), np.pi/4])

    # シミュレーションする環境の構築
    simulator = baumgarte2d.Simulator()

    # blockを追加する
    simulator.add_rigidbody(block)

    # (-l, 0)を(0, 0)へ一致拘束
    simulator.add_pinjoint_constrain(
        (-l, 0), block,
        (0, a*sin(nu*t)), None,
        dumper=c
    )

    # シミュレーション時間を設定しシミュレーション
    t_max = 5
    dt = 2*np.pi/nu/20
    ts = np.linspace(0, t_max, num=int(t_max/dt))
    xs = simulator.simulation(ts)

    # プロット
    plt.plot(ts, np.degrees(xs[:, 2]))
    plt.xlabel("time")
    plt.ylabel("$\\theta$")
    plt.show()

    plt.plot(xs[:, 0], xs[:, 1])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    # 描画
    if not os.path.exists(IMAGE_FILE_PATH):
        os.mkdir(IMAGE_FILE_PATH)
    simulator.draw_all(
        ts, xs, 4, xlim=[-3, +3], ylim=[-2, 2.5], save_format=IMAGE_FILE_PATH+"/%04d.png")

    print(simulator.calc_cq())
    print(simulator.calc_cqdotqq())
    print(simulator.calc_cqt())
    print(simulator.calc_ctt())
    print(simulator.calc_force())
    print(simulator.calc_mass())


if __name__ == "__main__":
    main()
