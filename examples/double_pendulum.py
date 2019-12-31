import baumgarte2d
import sympy
import numpy as np
import matplotlib.pyplot as plt


def main():
    l1 = sympy.symbols("l_1")
    l2 = sympy.symbols("l_2")
    g = sympy.symbols("g")
    c = sympy.symbols("c")

    # 剛体の定義
    block1 = baumgarte2d.RigidBody(1)
    block2 = baumgarte2d.RigidBody(2)

    # 重力の追加
    block1.add_force_y(0-block1.m*g)
    block2.add_force_y(0-block2.m*g)

    # 初期値の設定．順にx0, y0, theta0
    block1.initial_position = np.array([1/np.sqrt(2), 1/np.sqrt(2), np.pi/4])
    block2.initial_position = np.array([3/np.sqrt(2), 3/np.sqrt(2), np.pi/4])

    # シミュレーションする環境の構築
    simulator = baumgarte2d.Simulator()

    # blockを追加する
    simulator.add_rigidbody(block1)
    simulator.add_rigidbody(block2)

    # block1の(-l1, 0)をグローバル座標の(0, 0)へ一致拘束
    simulator.add_pinjoint_constrain(
        (0-l1, 0), block1,
        (0, 0), None,
        dumper=c
    )
    # block1の(l1, 0)をblock2の(-l2, 0)へ一致拘束
    simulator.add_pinjoint_constrain(
        (l1, 0), block1,
        (0-l2, 0), block2,
        dumper=c
    )

    # 行列を表示
    print(simulator.calc_cq())
    print(simulator.calc_cqdotqq())
    print(simulator.calc_cqt())
    print(simulator.calc_ctt())
    print(simulator.calc_force())
    print(simulator.calc_mass())

    # シミュレーション時間を設定しシミュレーション
    ts = np.linspace(0, 10, num=500)
    xs = simulator.simulation(
        ts, 10, 10, [(l1, 1), (l2, 1), (g, 9.8), (c, 0.05)])

    # プロット
    plt.plot(ts, np.degrees(xs[:, 2]), label="$\\theta_1$")
    plt.plot(ts, np.degrees(xs[:, 5]), label="$\\theta_2$")
    plt.xlabel("time")
    plt.ylabel("$\\theta$")
    plt.legend()
    plt.show()

    plt.plot(xs[:, 0], xs[:, 1], label="$(x_1,y_1)$")
    plt.plot(xs[:, 3], xs[:, 4], label="$(x_2,y_2)$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
