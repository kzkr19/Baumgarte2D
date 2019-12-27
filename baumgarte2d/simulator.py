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
