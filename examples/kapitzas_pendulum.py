import baumgarte2d
from sympy import symbols, sin
import numpy as np
import matplotlib.pyplot as plt


def main():
    a = 0.1
    l = 1.0
    g = 9.8
    c = 0.5
    # 振動の角速度がsqrt(2*g*l)/aより大きければ安定する
    # https://en.wikipedia.org/wiki/Kapitza%27s_pendulum
    nu = 2*np.sqrt(2*g*l)/a
    t = symbols("t")

    # 剛体の定義
    block = baumgarte2d.RigidBody(0)

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
    t_max = 20
    dt = 2*np.pi/nu/20
    ts = np.linspace(0, 10, num=int(t_max/dt))
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

    print(simulator.calc_cq())
    print(simulator.calc_cqdotqq())
    print(simulator.calc_cqt())
    print(simulator.calc_ctt())
    print(simulator.calc_force())
    print(simulator.calc_mass())


if __name__ == "__main__":
    main()
