import numpy as np
import sympy
from copy import deepcopy


class RigidBody:
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
        self.__sym_dot_x = sympy.symbols("\\dot{x_{%d}}" % object_id)
        self.__sym_dot_y = sympy.symbols("\\dot{y_{%d}}" % object_id)
        self.__sym_dot_theta = sympy.symbols("\\dot{\\theta_{%d}}" % object_id)
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
    def dot_x(self) -> sympy.Symbol:
        return deepcopy(self.__sym_dot_x)

    @property
    def dot_y(self) -> sympy.Symbol:
        return deepcopy(self.__sym_dot_y)

    @property
    def dot_theta(self) -> sympy.Symbol:
        return deepcopy(self.__sym_dot_theta)

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

    def convert_point(self, point0: sympy.Matrix, to_global=True) -> sympy.Matrix:
        """
        to_globalがTrueならローカル座標系の点をグローバル座標系の点にするメソッド
        Falseならグローバル座標系の点をローカル座標系の点に変換する

        point0: 変換対象の点
        to_global: Trueならpoint0をグローバル座標へ，Falseならローカル座標系へ変換する
        """

        if isinstance(point0, sympy.Matrix) == False:
            raise RuntimeError("point0 must be instance of sympy.Matrix.")

        if to_global:
            return self.position + self.rotation_matrix()*(point0)
        else:
            return self.rotation_matrix(False)*(point0 - self.position)

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
