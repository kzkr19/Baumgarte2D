import numpy as np
import sympy
from copy import deepcopy
from typing import Tuple, List, Union
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from typeguard import check_type
from .core import *


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
        self.__fx: sympy.Expr = 0
        self.__fy: sympy.Expr = 0
        self.__torque: sympy.Expr = 0

        # 初期値
        self.__initial_position: np.ndarray = np.zeros(3)
        self.__initial_velocity: np.ndarray = np.zeros(3)
        self.__mass: float = 1.0
        self.__moment_of_inertia: float = 1.0

        # 描画の設定
        self.__width: float = 1.0
        self.__height: float = 1.0
        self.__center_rect: Tuple[Number, Number] = (0, 0)
        self.__color: str = "black"

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
    def J(self) -> sympy.Symbol:
        return deepcopy(self.__sym_j)

    @property
    def position(self) -> sympy.Matrix:
        """
        グローバル座標から見た物体の原点
        """
        return sympy.Matrix([[self.x], [self.y]])

    @property
    def force_all(self) -> sympy.Matrix:
        return sympy.Matrix([[self.__fx, self.__fy, self.__torque]]).T

    @property
    def initial_position(self) -> np.ndarray:
        return self.__initial_position

    @initial_position.setter
    def initial_position(self, other: np.ndarray):
        self.__initial_position = other

    @property
    def initial_velocity(self) -> np.ndarray:
        return self.__initial_velocity

    @initial_velocity.setter
    def initial_velocity(self, other: np.ndarray):
        self.__initial_velocity = other

    @property
    def moment_of_inertia(self):
        return self.__moment_of_inertia

    @moment_of_inertia.setter
    def moment_of_inertia(self, other: float):
        self.__moment_of_inertia = other

    @property
    def mass(self):
        return self.__mass

    @mass.setter
    def mass(self, other: float):
        self.__mass = other

    @property
    def width(self):
        return self.__width

    @property
    def center_of_rectangle(self) -> Tuple[Number, Number]:
        return self.__center_rect

    @center_of_rectangle.setter
    def center_of_rectangle(self, other: Tuple[Number, Number]):
        check_type("center_of_rectangle", other, Tuple[Number, Number])
        self.__center_rect = other

    @width.setter
    def width(self, val: Number):
        check_type("width", val, Number)
        self.__width = val

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, val: Number):
        check_type("height", val, Number)
        self.__height = val

    @property
    def color(self) -> str:
        return self.__color

    @color.setter
    def color(self, other: str):
        check_type("color", other, str)
        self.__color = other

    def get_parameters(self) -> List[Tuple[sympy.Symbol, float]]:
        return [(self.J, self.moment_of_inertia), (self.m, self.mass)]

    def add_force_x(self, fx: sympy.Expr):
        self.__fx += fx

    def add_force_y(self, fy: sympy.Expr):
        self.__fy += fy

    def add_torque(self, torque: sympy.Expr):
        self.__torque += torque

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

    def convert_vector(self, vector: sympy.Matrix, to_global=True) -> sympy.Matrix:
        """
        to_globalがTrueならローカル座標系のベクトルをグローバル座標系のベクトルにするメソッド
        Falseならグローバル座標系のベクトルをローカル座標系のベクトルに変換する

        vector: 変換対象の点
        to_global: Trueならpoint0をグローバル座標へ，Falseならローカル座標系へ変換する
        """
        if isinstance(vector, sympy.Matrix) == False:
            raise RuntimeError("point0 must be instance of sympy.Matrix.")

        return self.rotation_matrix(to_global)*vector

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

    def calc_points_local(self) -> List[Tuple[Number, Number]]:
        """
        ローカル座標系での四角形の4頂点の座標を返すメソッド
        """
        dx, dy = self.center_of_rectangle
        w2, h2 = self.width/2, self.height/2

        points_l = [(dx-w2, dy-h2), (dx+w2, dy-h2),
                    (dx+w2, dy+h2), (dx-w2, dy+h2)]

        return points_l

    def calc_points_global(self, x: float, y: float, theta: float) -> List[Tuple[float, float]]:
        """
        グローバル座標系での四角形の4頂点の座標を返すメソッド

        x: 現在の重心のx座標
        y: 現在の重心のy座標
        theta: 現在の剛体の角度[rad]
        """
        check_type("x", x, float)
        check_type("y", y, float)
        check_type("theta", theta, float)

        points_l = self.calc_points_local()

        c = np.cos(theta)
        s = np.sin(theta)
        ret = []
        for p in points_l:
            xx = x + c * p[0] - s * p[1]
            yy = y + s * p[0] + c * p[1]
            ret.append((xx, yy))
        return ret

    def draw(self, ax: Axes, x: float, y: float, theta: float):
        """
        x: 重心座標[m]
        y: 重心座標[m]
        theta: 回転角度[rad]
        """
        points_g = self.calc_points_global(x, y, theta)
        p = Polygon(points_g, color=self.color)
        ax.add_patch(p)
