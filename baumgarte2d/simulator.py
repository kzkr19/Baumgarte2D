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
        # 速度の変数
        return sum([[r.dot_x, r.dot_y, r.dot_theta] for r in self.__bodies], [])

    def calc_c(self) -> sympy.Matrix:
        return sympy.Matrix([self.__constrains]).T

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

    def calc_cqdotqq(self) -> sympy.Matrix:
        """
        \\frac{\\partial C_q \\dot{q}}{\\partial q}を計算するメソッド
        """
        # 位置の変数の数
        n_var = len(self.__bodies) * 3
        # 制約の数
        n_constrain = len(self.__constrains)

        cq = self.calc_cq()
        cq_dot_q = cq*sympy.Matrix([self.dot_q]).T
        def f(i, j): return sympy.diff(cq_dot_q[i], self.q[j])

        return sympy.Matrix(n_constrain, n_var, f)

    def calc_cqt(self) -> sympy.Matrix:
        t = sympy.symbols("t")
        return sympy.diff(self.calc_cq(), t)

    def calc_ct(self) -> sympy.Matrix:
        t = sympy.symbols("t")
        return sympy.diff(self.calc_cq(), t)

    def calc_ctt(self) -> sympy.Matrix:
        t = sympy.symbols("t")
        return sympy.diff(self.calc_cq(), t, 2)

    def calc_mass(self) -> sympy.Matrix:
        Mass = sum([[r.m, r.m, r.J] for r in self.__bodies], [])
        Mass = sympy.diag(*Mass)
        return Mass

    def calc_force(self) -> sympy.Matrix:
        xs = sum([[x.force_all[0], x.force_all[1], x.force_all[2]]
                  for x in self.__bodies], [])
        return sympy.Matrix([xs]).T

    def calc_gamma(self, alpha, beta) -> sympy.Matrix:
        dot_q = self.dot_q
        c = self.calc_c()
        cq = self.calc_cq()
        ct = self.calc_ct()
        ctt = self.calc_ctt()
        cqt = self.calc_cqt()
        cqdotqq = self.calc_cqdotqq()
        gamma = -cqdotqq * dot_q
        gamma += -2 * alpha * (cq * dot_q + ct)
        gamma += -2 * cqt * dot_q
        gamma += -ctt
        gamma += -beta*beta*c
        return gamma

    def calc_dotdotq(self, alpha, beta) -> sympy.Matrix:
        n_body = len(self.__bodies)
        n_constrain = len(self.__constrains)
        cq = self.calc_cq()
        force = self.calc_force()
        gamma = self.calc_gamma(alpha, beta)
        mass = self.calc_mass()

        mat_left = sympy.BlockMatrix(
            [[mass, cq.T], [cq, sympy.ZeroMatrix(n_constrain, n_body*3)]])
        mat_right = sympy.BlockMatrix([[force], [gamma]])

        result = mat_left.inv()*mat_right
        return result[:n_body*3]
