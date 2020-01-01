import numpy as np
import sympy
from typing import Tuple, Union, List
from .rigidbody import RigidBody
from functools import reduce
from scipy.integrate import odeint


class Simulator:
    def __init__(self):
        self.__constrains: List[sympy.Expr] = []
        self.__bodies: List[RigidBody] = []

    def add_rigidbody(self, rigidbody: RigidBody):
        """
        シミュレーション対象にrigidbodyを追加するメソッド
        """
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

    def add_pinjoint_constrain(
            self,
            p1: Tuple[sympy.Expr, sympy.Expr],
            b1: RigidBody,
            p2: Tuple[sympy.Expr, sympy.Expr],
            b2: RigidBody = None,
            dumper: sympy.Expr = None,
            spring: sympy.Expr = None):
        """
        剛体b1のローカル座標系の点p1と，
        剛体b2のローカル座標系の点p2を一致拘束させる拘束を追加するメソッド．

        p1: 剛体b1のローカル座標系の点
        b1: 剛体b1
        p2: 剛体b2のローカル座標系の点．b2=Noneの場合グローバル座標系となる
        b2: 剛体b2．Noneの場合はp2がグローバル座標系の点となる
        dumper: トーションダンパー係数[Ns/rad]．Noneなら無し
        spring: トーションスプリング係数[N/rad]．Noneなら無し
        """
        # p1, p2をグローバル座標系にする
        p1 = sympy.Matrix([p1]).T
        p2 = sympy.Matrix([p2]).T
        p1 = b1.convert_point(p1)
        p2 = p2 if b2 is None else b2.convert_point(p2)

        # 角度/角速度の差
        delta_theta = b1.theta - (0 if b2 is None else b2.theta)
        delta_dtheta = b1.dot_theta - (0 if b2 is None else b2.dot_theta)

        # ダンピングの追加
        if dumper is not None:
            torque = -delta_dtheta*dumper
            b1.add_torque(torque)
            if b2 is not None:
                b2.add_torque(-torque)

        # バネの追加
        # TODO: 45度を中心としてトルクばねを設定したい場合
        if spring is not None:
            torque = -delta_theta*spring
            b1.add_torque(torque)
            if b2 is not None:
                b2.add_torque(torque)

        # 制約の追加
        constrain = p1 - p2
        self.add_constrain(constrain[0])
        self.add_constrain(constrain[1])

    def add_slide_constrain(
        self,
        v1: Tuple[sympy.Expr, sympy.Expr],
        p1: Tuple[sympy.Expr, sympy.Expr],
        b1: RigidBody,
        v2: Tuple[sympy.Expr, sympy.Expr],
        p2: Tuple[sympy.Expr, sympy.Expr],
        b2: RigidBody = None,
    ):
        """
        d=(p1-p2)とv1，v1とv2を並行にする並進拘束を追加するメソッド

        v1: 剛体b1のローカル座標系で定義されたベクトル
        p1: 剛体b1のローカル座標系の点
        b1: 剛体b1のRigidBodyオブジェクト
        v2: 剛体b2のローカル座標系で定義されたベクトル．b2=Noneの場合グローバル座標系となる
        p2: 剛体b2のローカル座標系の点．b2=Noneの場合グローバル座標系となる
        b2: 剛体b2．Noneの場合はp2, v2がグローバル座標系の点となる
        """
        v1 = sympy.Matrix([v1]).T
        v2 = sympy.Matrix([v2]).T
        p1 = sympy.Matrix([p1]).T
        p2 = sympy.Matrix([p2]).T

        # グローバル座標へ変換
        v1 = b1.convert_vector(v1)
        v2 = v2 if b2 is None else b2.convert_vector(v2)
        p1 = b1.convert_point(p1)
        p2 = p2 if b2 is None else b2.convert_vector(p2)
        d = p1-p2

        rot_90 = sympy.Matrix([[0, -1], [1, 0]])
        vertical_v1 = rot_90*v1

        constrain_1 = vertical_v1.T * d
        constrain_2 = vertical_v1.T * v2

        self.add_constrain(constrain_1[0])
        self.add_constrain(constrain_2[0])

    def get_body_parameters(self) -> List[Tuple[sympy.Symbol, float]]:
        """
        追加された剛体の定数の値を取得するメソッド
        """
        return sum([r.get_parameters() for r in self.__bodies], [])

    def get_initial_position(self) -> np.ndarray:
        """
        全剛体の初期位置のリストを返すメソッド
        """
        return reduce(lambda x, y: np.r_[x, y], [r.initial_position for r in self.__bodies])

    def get_initial_velocity(self) -> np.ndarray:
        """
        全剛体の初期速度のリストを返すメソッド
        """
        return reduce(lambda x, y: np.r_[x, y], [r.initial_velocity for r in self.__bodies])

    def calc_c(self) -> sympy.Matrix:
        """
        制約条件C=0なるCを求めるメソッド
        """
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
        """
        C_qをtで偏微分した行列を計算するメソッド
        """
        t = sympy.symbols("t")
        return sympy.diff(self.calc_cq(), t)

    def calc_ct(self) -> sympy.Matrix:
        """
        Cをtで偏微分した行列を計算するメソッド
        """
        t = sympy.symbols("t")
        return sympy.diff(self.calc_c(), t)

    def calc_ctt(self) -> sympy.Matrix:
        """
        Cをtで2回偏微分した行列を計算するメソッド
        """
        t = sympy.symbols("t")
        return sympy.diff(self.calc_c(), t, 2)

    def calc_mass(self) -> sympy.Matrix:
        """
        質量マトリクスを計算するメソッド
        """
        Mass = sum([[r.m, r.m, r.J] for r in self.__bodies], [])
        Mass = sympy.diag(*Mass)
        return Mass

    def calc_force(self) -> sympy.Matrix:
        """
        各剛体にかかる外力をまとめたベクトルを計算するメソッド
        """
        xs = sum([[x.force_all[0], x.force_all[1], x.force_all[2]]
                  for x in self.__bodies], [])
        return sympy.Matrix([xs]).T

    def calc_gamma(self, alpha: float = 10, beta: float = 10) -> sympy.Matrix:
        """
        制約を常に満たすような拘束力を計算するメソッド

        alpha: バウムガルテの安定化法の減衰係数
        beta: バウムガルテの安定化法のばね定数
        """
        dot_q = sympy.Matrix([self.dot_q]).T
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

    def calc_dotdotq_equation(self, alpha: float = 10, beta: float = 10) -> sympy.Matrix:
        """
        状態変数の2回微分を計算する連立方程式
        mat_left * {dotdot_q, lambdas} = mat_right
        の行列mat_left, mat_rightの解析解を計算するメソッド

        alpha: バウムガルテの安定化法の減衰係数
        beta: バウムガルテの安定化法のばね定数
        """
        n_body = len(self.__bodies)
        n_constrain = len(self.__constrains)
        cq = self.calc_cq()
        force = self.calc_force()
        gamma = self.calc_gamma(alpha, beta)
        mass = self.calc_mass()

        # 連立方程式の構築．mat_left * {dotdot_q, lambdas} = mat_right
        mat_left = sympy.BlockMatrix(
            [[mass, cq.T], [cq, sympy.ZeroMatrix(n_constrain, n_body*3)]])
        mat_left = sympy.Matrix(mat_left)
        mat_right = sympy.BlockMatrix([[force], [gamma]])
        mat_right = sympy.Matrix(mat_right)

        return mat_left, mat_right

    def simulation(
            self,
            ts: np.ndarray,
            alpha: float = 10,
            beta: float = 10,
            parameters: List[Tuple[sympy.Symbol, float]] = None):
        """
        シミュレーションを実行するメソッド

        ts: シミュレーションしたい時間のリスト
        alpha: バウムガルテの安定化法の減衰係数
        beta: バウムガルテの安定化法のばね定数
        parameters: ユーザが独自定義した定数とその値のタプルのリスト
        """
        n_body = len(self.__bodies)
        # 求解する変数のシンボル
        q = self.q
        dot_q = self.dot_q
        t = sympy.symbols("t")

        # dot_q(速度)の二階微分を表す行列(解析解)
        print("calculating analytical solution...")
        mat_left, mat_right = self.calc_dotdotq_equation(alpha, beta)

        # parametersへ質量などの値を追加する
        if parameters is None:
            parameters = []
        parameters += self.get_body_parameters()

        # 定数を代入し，あとは変数q, dot_qを計算すれば良い状態にする
        mat_left = mat_left.subs(parameters)
        mat_right = mat_right.subs(parameters)

        # Cにコンパイルできるように変数名を変更する
        print("compiling code...")
        original_variables = q + dot_q
        new_variables = [sympy.symbols("x%d" % i)
                         for i in range(len(original_variables))]
        var_names = list(zip(original_variables, new_variables))
        mat_left = mat_left.subs(var_names)
        mat_right = mat_right.subs(var_names)

        # Cへコンパイル
        # TODO: CodeGenArgumentListErrorの対処
        from sympy.utilities.autowrap import autowrap
        calc_mat_left = autowrap(
            mat_left,
            args=[t] + new_variables,
            language="C", backend="cython")
        calc_mat_right = autowrap(
            mat_right,
            args=[t] + new_variables,
            language="C", backend="cython")

        print("calculating numerical solution...")

        def dxdt(x, t):
            """
            時刻t，状態変数がxのときのxの時間微分を返す関数
            """
            vel = x[n_body*3:]
            mat_left = calc_mat_left(t, *x)
            mat_right = calc_mat_right(t, *x)[:, 0]
            d_vel = np.linalg.solve(mat_left, mat_right)[:n_body*3]

            return np.r_[vel, d_vel.astype(np.float)]

        x0 = np.r_[self.get_initial_position(), self.get_initial_velocity()]
        return odeint(dxdt, x0, ts)
