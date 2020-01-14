from baumgarte2d import RigidBody, Simulator
from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from copy import deepcopy
import os

IMAGE_FILE_PATH: str = "./examples/temp"


def main():
    # 記号・値の定義
    t = symbols("t")
    sym_ls = [symbols("l_%d" % i) for i in range(5)]
    val_ls = [0.5, 1.0, 1.0, 1.0, 1.0]
    sym_g, val_g = symbols("g"), 9.8
    sym_omega, val_omega = symbols("\\omega"), 20
    sym_cs = [symbols("c_%d" % i) for i in range(2)]
    val_cs = [0.05, 0.05]
    params = list(zip(sym_ls, val_ls))
    params += list(zip(sym_cs, val_cs))
    params += [(sym_g, val_g), (sym_omega, val_omega)]

    # 剛体の定義
    blocks: List[RigidBody] = [RigidBody(i) for i in range(5)]

    # シミュレーションする環境の構築
    simulator = Simulator()

    # 初期値の設定．順にx0, y0, theta0
    blocks[0].initial_position = np.array([0, val_ls[0], 0])
    blocks[1].initial_position = np.array([
        0,
        blocks[0].initial_position[1] + val_ls[0] + val_ls[1],
        0
    ])
    blocks[2].initial_position = np.array([
        0,
        blocks[1].initial_position[1] + val_ls[1] + val_ls[2],
        0
    ])
    blocks[3].initial_position = np.array([
        val_ls[3]*np.cos(np.pi/4),
        blocks[2].initial_position[1] + val_ls[2] + val_ls[3]*np.sin(np.pi/4),
        -np.pi/4
    ])
    blocks[4].initial_position = np.array([
        blocks[3].initial_position[0]
        + (val_ls[3] + val_ls[4])*np.cos(np.pi/4),
        blocks[3].initial_position[1]
        + (val_ls[3] + val_ls[4])*np.sin(np.pi/4),
        -np.pi/4
    ])

    for i in range(5):
        # 重力の追加
        blocks[i].add_force_y(0-blocks[i].m*sym_g)

        # 剛体の追加
        simulator.add_rigidbody(blocks[i])

        # 描画時の設定
        blocks[i].height = val_ls[i]*2
        blocks[i].width = 0.2
        blocks[i].color = ["red", "green",
                           "blue", "pink", "black", "orange"][i]

    # 拘束の追加
    # 原点に剛体0を拘束
    simulator.add_pinjoint_constrain(
        (0, -sym_ls[0]), blocks[0],
        (0, 0), None
    )
    # すべての剛体をピンジョイントで拘束する
    for i in range(4):
        simulator.add_pinjoint_constrain(
            (0, +sym_ls[i+0]), blocks[i+0],
            (0, -sym_ls[i+1]), blocks[i+1]
        )
    # 剛体0の角度を運動拘束する
    simulator.add_constrain(blocks[0].theta-sym_omega*t)
    # 剛体2を並進拘束する
    # simulator.add_slide_constrain(
    #    (1, 0), (0, 0), blocks[2],
    #    (0, 1), (0, 0), None
    # )
    simulator.add_constrain(
        blocks[2].theta
    )
    simulator.add_constrain(
        blocks[2].x
    )

    # 行列を表示
    # print(simulator.calc_cq())
    # print(simulator.calc_cqdotqq())
    # print(simulator.calc_cqt())
    # print(simulator.calc_ctt())
    # print(simulator.calc_force())
    # print(simulator.calc_mass())

    # シミュレーション時間を設定しシミュレーション
    dt = (2*np.pi/val_omega)/20.0
    tmax = 10
    num = int(tmax/dt)
    ts = np.linspace(0, tmax, num=num)
    xs = simulator.simulation(ts, parameters=params)

    # プロット
    # for i in range(len(blocks)):
    #     plt.plot(xs[:, 3*i+0], xs[:, 3*i+1], label="$(x_%d,y_%d)$" % (i, i))
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.legend()
    # plt.show()

    if not os.path.exists(IMAGE_FILE_PATH):
        os.mkdir(IMAGE_FILE_PATH)
    simulator.draw_all(ts, xs, 1,
                       xlim=(-8, +8),
                       ylim=(-2, +10),
                       save_format=IMAGE_FILE_PATH+"/%04d.png")


if __name__ == "__main__":
    main()
