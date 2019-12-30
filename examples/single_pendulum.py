import baumgarte2d
import sympy
import numpy as np
import matplotlib.pyplot as plt


def main():
    l = sympy.symbols("l")
    g = sympy.symbols("g")

    # 剛体の定義
    block = baumgarte2d.RigidBody(0)

    # 重力の追加
    block.add_force(sympy.Matrix([[0, 0-block.m*g, 0]]).T)

    # 初期値の設定．順にx0, y0, theta0
    block.initial_position = np.array([1/np.sqrt(2), -1/np.sqrt(2), -np.pi/4])

    # [-l, 0]のグローバル座標を取得
    block_tip_pos = block.convert_point(sympy.Matrix([[0-l], [0]]))

    # シミュレーションする環境の構築
    simulator = baumgarte2d.Simulator()

    # blockを追加する
    simulator.add_rigidbody(block)

    # 常に0となるべき制約式の追加
    simulator.add_constrain(block_tip_pos[0])
    simulator.add_constrain(block_tip_pos[1])

    # シミュレーション時間を設定しシミュレーション
    ts = np.linspace(0, 10, num=100)
    xs = simulator.simulation(ts, 10, 10, [(l, 1), (g, 9.8)])

    # プロット
    plt.plot(ts, np.degrees(xs[:, 2]))
    plt.xlabel("time")
    plt.ylabel("$\\theta$")
    plt.show()

    plt.plot(xs[:, 0], xs[:, 1])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


if __name__ == "__main__":
    main()
