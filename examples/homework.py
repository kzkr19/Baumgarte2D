from baumgarte2d import RigidBody, Simulator
from sympy import symbols, latex
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from copy import deepcopy
import os

IMAGE_FILE_PATH: str = "./examples/temp"


def do_simulation(omega: float, save_image: bool):
    # 記号・値の定義
    t = symbols("t")
    sym_ls = [symbols("l_%d" % i) for i in range(5)]
    val_ls = [0.1, 0.5, 0.5, 0.3, 0.3]
    sym_g, val_g = symbols("g"), 9.8
    sym_c, val_c = symbols("c"), 0.1
    sym_omega, val_omega = symbols("\\omega"), omega
    params = list(zip(sym_ls, val_ls))
    params += [(sym_c, val_c)]
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
        blocks[i].width = 0.02
        blocks[i].color = ["red", "green",
                           "blue", "pink", "orange"][i]
        blocks[i].mass = 0.125 * val_ls[i]*2
        blocks[i].moment_of_inertia = (4/3)*blocks[i].mass*val_ls[i]**2

    # 拘束の追加
    # すべての剛体をピンジョイントで拘束する
    for i in range(4):
        simulator.add_pinjoint_constrain(
            (0, +sym_ls[i+0]), blocks[i+0],
            (0, -sym_ls[i+1]), blocks[i+1],
            dumper=sym_c
        )
    # 原点に剛体0を拘束
    simulator.add_pinjoint_constrain(
        (0, -sym_ls[0]), blocks[0],
        (0, 0), None
    )
    # 剛体2を並進拘束する
    simulator.add_slide_constrain(
        (0, 1), (0, 0), blocks[2],
        (0, 1), (0, 0), None
    )
    # 剛体0の角度を運動拘束する
    simulator.add_constrain(blocks[0].theta-sym_omega*t)

    # シミュレーション時間を設定しシミュレーション
    dt = (2*np.pi/val_omega)/20.0
    tmax = 10
    num = int(tmax/dt)
    ts = np.linspace(0, tmax, num=num)
    xs, forces = simulator.simulation(ts, parameters=params, return_force=True)

    if save_image:
        if not os.path.exists(IMAGE_FILE_PATH):
            os.mkdir(IMAGE_FILE_PATH)
        simulator.draw_all(ts, xs, 1,
                           xlim=(-10/3, +10/3),
                           ylim=(-1, +4),
                           save_format=IMAGE_FILE_PATH+"/%04d.png")

    return ts, xs, forces


def main():
    # 使ってみるomegaのリスト
    omega_list: List[float] = [10.0, 20.0, 40.0, 80.0]
    xs_list = []
    ts_list = []
    force_list = []

    for omega in omega_list:
        # omegaを変えてシミュレーションする
        ts, xs, forces = do_simulation(omega, False)
        xs_list.append(xs)
        ts_list.append(ts)
        force_list.append(forces)

    # 二重振り子の角度をプロットする
    plt.figure()
    axe1 = plt.subplot(2, 1, 1)
    axe2 = plt.subplot(2, 1, 2)
    for omega, xs, ts in zip(omega_list, xs_list, ts_list):
        axe1.plot(ts, np.degrees(xs[:, 3*3+2]),
                  label="$\\omega=%5.1lf$ rad/s" % omega)
        axe2.plot(ts, np.degrees(xs[:, 3*4+2]),
                  label="$\\omega=%5.1lf$ rad/s" % omega)
    axe1.legend()
    axe2.legend()
    axe1.set_title("body 3")
    axe2.set_title("body 4")
    plt.ylabel("Angle [Degree]")
    plt.xlabel("Time [s]")
    plt.show()

    # モーターのトルクをプロットする
    plt.figure()
    for omega, ts, forces in zip(omega_list, ts_list, force_list):
        forces = np.array(forces)
        plt.plot(ts, forces[:, 0*3+2], label="$\\omega=%5.1lf$ rad/s" % omega)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Torque[Nm]")
    plt.show()


if __name__ == "__main__":
    main()
