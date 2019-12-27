import numpy as np
import sympy
from typing import Tuple, Union, List
from .rigidbody import RigidBody


class Simulator:
    def __init__(self):
        self.__constrains: List[sympy.Expr] = []
        self.__bodies: List[RigidBody] = []

    def add_rigidbody(self, rigidbody: RigidBody):
        if isinstance(rigidbody, RigidBody):
            self.__bodies.append(rigidbody)
        else:
            raise RuntimeError("Unkown type: %s" % type(rigidbody))

    def add_constrain(self, constrain: Union[sympy.Expr, List[sympy.Expr]]):
        """
        制約条件を加えるメソッド
        """
        if isinstance(constrain, list):
            for c in constrain:
                self.add_constrain(c)
        elif isinstance(constrain, sympy.Expr):
            self.__constrains.append(constrain)
        else:
            raise RuntimeError("Unkown type: %s" % type(constrain))

    @property
    def q(self) -> List[sympy.Expr]:
        """
        追加された剛体の位置の変数(x, y, theta)のリストを返すメソッド
        """
        # 位置の変数
        return sum([[r.x, r.y, r.theta] for r in self.__bodies], [])

    @property
    def dot_q(self) -> List[sympy.Expr]:
        """
        追加された剛体の位置の変数(dx/dt, dy/dt, d\\theta/dt)のリストを返すメソッド
        """
        t = sympy.symbols("t")
        q = self.q
        return [sympy.Derivative(x, t) for x in q]

    def calc_cq(self) -> sympy.Matrix:
        """
        拘束式 C=0 について，Cを位置変数qで偏微分した行列C_qを返すメソッド
        C_{q,i,j}=\\frac{\\partial C_i}{\\partial q_j}
        """
        # 位置の変数の数
        n_var = len(self.__bodies) * 3
        # 制約の数
        n_constrain = len(self.__constrains)

        # Cq(i,j)の値を計算する関数
        def f(i, j): return sympy.diff(self.__constrains[i], self.q[j])

        return sympy.Matrix(n_constrain, n_var, f)
