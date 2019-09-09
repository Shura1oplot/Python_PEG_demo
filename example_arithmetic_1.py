#!/usr/bin/env python3

import sys
from peg import *


class ArithmeticGrammar(Grammar):

    Start = Reference("Additive")

    Additive = Choice(
        Action("add", Sequence(
            Label("left", Reference("Multiplicative")),
            MatchString("+"),
            Label("right", Reference("Additive"))
        )),
        Reference("Multiplicative"),
    )

    @staticmethod
    def _act_add(ctx):
        return ctx.left + ctx.right

    Multiplicative = Choice(
        Action("mul", Sequence(
            Label("left", Reference("Primary")),
            MatchString("*"),
            Label("right", Reference("Multiplicative"))
        )),
        Reference("Primary"),
    )

    @staticmethod
    def _act_mul(ctx):
        return ctx.left * ctx.right

    Primary = Choice(
        Reference("Integer"),
        Action("additive", Sequence(
            MatchString("("),
            Label("additive", Reference("Additive")),
            MatchString(")"),
        ))
    )

    @staticmethod
    def _act_additive(ctx):
        return ctx.additive

    Integer = As(
        "Integer",
        Action("digits", Label("digits", Text(Some(MatchClass("0-9")))))
    )

    @staticmethod
    def _act_digits(ctx):
        return int(ctx.digits)


def main(argv=sys.argv):
    parser = ArithmeticGrammar()
    print(parser)
    print(parser("2*(3+4)"))


if __name__ == "__main__":
    sys.exit(main())
