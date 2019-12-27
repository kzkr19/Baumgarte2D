import numpy as np
import sympy
from copy import deepcopy


class BlockObject:
    def __init__(self, object_id: int):
        """

        mass: 質量[kg]
        inertia: 慣性モーメント[kg m^2]
        """

        self.__id: int = object_id

        # 記号の定義
        self.__sym_x = sympy.symbols("x_{%d}" % object_id)
        self.__sym_y = sympy.symbols("y_{%d}" % object_id)
        self.__sym_theta = sympy.symbols("\\theta_{%d}" % object_id)
        self.__sym_m = sympy.symbols("m_{%d}" % object_id)
        self.__sym_j = sympy.symbols("J_{%d}" % object_id)

        # 拘束力を除く外力
        self.__force_all = sympy.Matrix([[0], [0], [0]])

    @property
    def x(self) -> sympy.Symbol:
        return deepcopy(self.__sym_x)

    @property
    def y(self) -> sympy.Symbol:
        return deepcopy(self.__sym_y)

    @property
    def theta(self) -> sympy.Symbol:
        return deepcopy(self.__sym_theta)

    @property
    def m(self) -> sympy.Symbol:
        return deepcopy(self.__sym_m)

    @property
    def inertia(self) -> sympy.Symbol:
        return deepcopy(self.__sym_j)

    @property
    def position(self) -> sympy.Matrix:
        """
        グローバル座標から見た物体の原点
        """
        return sympy.Matrix([[self.x], [self.y]])

    @property
    def force_all(self) -> sympy.Matrix:
        return self.__force_all

    def rotation_matrix(self, local_to_global=True):
        """
        回転行列を返すメソッド

        local_to_global: 
            Trueならこのオブジェクトのローカル座標のベクトルをグローバル座標のベクトルに
            Falseならその逆行列を返すメソッド
        """
        rot = sympy.Matrix([
            [sympy.cos(self.theta), -sympy.sin(self.theta)],
            [sympy.sin(self.theta), sympy.cos(self.theta)]
        ])

        return rot if local_to_global else rot.T
