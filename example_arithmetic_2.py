#!/usr/bin/env python3

import sys
from peg import *


class ArithmeticGrammar(Grammar):

    __grammar__ = """
        Start
          = Additive

        Additive
          = left:Multiplicative "+" right:Additive {add}
          / Multiplicative

        Multiplicative
          = left:Primary "*" right:Multiplicative {mul}
          / Primary

        Primary
          = Integer
          / "(" additive:Additive ")" {additive}

        Integer "Integer"
          = digits:$[0-9]+ {digits}
    """

    @staticmethod
    def _act_add(ctx):
        return ctx.left + ctx.right

    @staticmethod
    def _act_mul(ctx):
        return ctx.left * ctx.right

    @staticmethod
    def _act_additive(ctx):
        return ctx.additive

    @staticmethod
    def _act_digits(ctx):
        return int(ctx.digits)


def main(argv=sys.argv):
    parser = ArithmeticGrammar()
    print(parser)
    print(parser("2*(3+4)"))


if __name__ == "__main__":
    sys.exit(main())
