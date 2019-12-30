import baumgarte2d
import sympy
import numpy as np
import matplotlib.pyplot as plt


def main():
    l = sympy.symbols("l")
    g = sympy.symbols("g")
    c = sympy.symbols("c")
    k = sympy.symbols("k")

    # 剛体の定義
    block = baumgarte2d.RigidBody(0)

    # 重力の追加
    block.add_force_y(0-block.m*g)

    # 初期値の設定．順にx0, y0, theta0
    block.initial_position = np.array([1/np.sqrt(2), -1/np.sqrt(2), -np.pi/4])

    # シミュレーションする環境の構築
    simulator = baumgarte2d.Simulator()

    # blockを追加する
    simulator.add_rigidbody(block)

    # (-l, 0)を(0, 0)へ一致拘束
    simulator.add_pinjoint_constrain(
        (0-l, 0), block,
        (0, 0), None,
        c, k
    )

    # シミュレーション時間を設定しシミュレーション
    ts = np.linspace(0, 10, num=100)
    xs = simulator.simulation(ts, 10, 10, [(l, 1), (g, 9.8), (c, 0.0), (k, 0)])

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
