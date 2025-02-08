import operator
from dataclasses import dataclass
from typing import Callable, Optional, Union

import flax.linen as nn


@dataclass
class ExpressionNode:
    """Represents a computation tree for module compositions with correct precedence."""

    left: Union[nn.Module, float, "ExpressionNode"]
    op: Optional[Callable] = None
    right: Optional[Union[nn.Module, float, "ExpressionNode"]] = None

    def evaluate(self, x, variables):
        """Recursively evaluates the computation tree while supporting parameter updates."""
        if self.op is None:
            return (
                self.left.apply(variables, x)
                if isinstance(self.left, nn.Module)
                else self.left
            )

        left_val = (
            self.left.evaluate(x, variables)
            if isinstance(self.left, ExpressionNode)
            else self.left.apply(variables, x)
        )
        right_val = (
            self.right.evaluate(x, variables)
            if isinstance(self.right, ExpressionNode)
            else self.right.apply(variables, x)
            if isinstance(self.right, nn.Module)
            else self.right
        )

        return self.op(left_val, right_val)

    def __add__(self, other):
        return ExpressionNode(self, operator.add, other)

    def __sub__(self, other):
        return ExpressionNode(self, operator.sub, other)

    def __mul__(self, other):
        return ExpressionNode(self, operator.mul, other)

    def __truediv__(self, other):
        return ExpressionNode(self, operator.truediv, other)

    def __matmul__(self, other):
        return ExpressionNode(self, operator.matmul, other)

    def __neg__(self):
        return ExpressionNode(self, operator.mul, -1)

    def __rmul__(self, other):
        return ExpressionNode(self, operator.mul, other)


class ComposableModule(nn.Module):
    """A Flax module that supports arithmetic operations."""

    expr: Optional[ExpressionNode] = None

    def __call__(self, x):
        return self.expr.evaluate(x, self.variables)

    def _binary_op(self, other, op):
        return ComposableModule(
            expr=ExpressionNode(
                self.expr if self.expr is not None else self,
                op,
                other.expr if isinstance(other, ComposableModule) else other,
            )
        )

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __mul__(self, other):
        return self._binary_op(other, operator.mul)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv)

    def __matmul__(self, other):
        return self._binary_op(other, operator.matmul)

    def __neg__(self):
        return self._binary_op(-1, operator.mul)

    def __rmul__(self, other):
        return self._binary_op(other, operator.mul)
