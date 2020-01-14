import baumgarte2d
import sympy
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_FILE_PATH: str = "./examples/temp"


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

    # 描画の設定
    block.width = 2.0
    block.height = 0.02
    block.color = "red"

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
    ts = np.linspace(0, 10, num=500)
    xs = simulator.simulation(ts, 10, 10, [(l, 1), (g, 9.8), (c, 0.5), (k, 0)])

    # 描画
    if not os.path.exists(IMAGE_FILE_PATH):
        os.mkdir(IMAGE_FILE_PATH)
    simulator.draw_all(ts, xs, 10, save_format=IMAGE_FILE_PATH+"/%04d.png")

    print(simulator.calc_cq())
    print(simulator.calc_cqdotqq())
    print(simulator.calc_cqt())
    print(simulator.calc_ctt())
    print(simulator.calc_force())
    print(simulator.calc_mass())


if __name__ == "__main__":
    main()
