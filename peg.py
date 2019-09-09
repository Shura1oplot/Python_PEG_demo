# [SublimeLinter @python:3]

from collections import OrderedDict
from collections.abc import Iterator
from locale import getpreferredencoding
import io
import re


class NothingType(Iterator):

    __slots__ = ()

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    def __init__(self):
        super().__init__()

    def __next__(self):
        raise StopIteration()

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __str__(self):
        return ""

    def __repr__(self):
        return "Nothing"


Nothing = NothingType()


class Marker(str):

    def __repr__(self):
        return str(self)


ANY = Marker("ANY")
EOI = Marker("EOI")


def quote(s):
    if isinstance(s, Marker):
        return s

    c = "'" if "'" not in s or '"' in s else '"'
    escape_map = {"\0": "\\0", "\a": "\\a", "\b": "\\b", "\t": "\\t",
                  "\n": "\\n", "\v": "\\v", "\f": "\\f", "\r": "\\r",
                  "\\": "\\\\"}
    escape_map[c] = "\\{}".format(c)
    return "".join((c, "".join(escape_map.get(c, c) for c in s), c))


class GrammarError(Exception):
    pass


class ParseError(Exception):
    pass


class MatchFailed(ParseError):

    def __init__(self, position, expected, found):
        super().__init__()

        self.position = position
        self.coordinates = None
        self.expected = set(expected)
        self.found = found

    def __str__(self):
        expected = self.expected

        if len(expected) > 1:
            lst = sorted(expected, key=lambda x: (isinstance(x, Marker), x),
                         reverse=True)
            expected = "{} or {}".format(", ".join(lst[:-1]), lst[-1])

        else:
            expected = next(iter(expected))

        err = "{} expected but {} found".format(expected, quote(self.found))

        if not self.coordinates:
            return "Position {}: {}".format(self.position, err)

        template = "Line {0[0]}, Column {0[1]}: {1}"
        return template.format(self.coordinates, err)


class ContextError(ParseError):
    pass


class ContextNameError(ContextError):

    def __init__(self, name):
        super().__init__()

        self.name = name

    def __str__(self):
        return "name {} is not defined".format(quote(self.name))


class ContextRoot(object):

    def __init__(self):
        super().__init__()

    def __getattr__(self, name):
        raise ContextNameError(name)


class Context(ContextRoot):

    def __init__(self, parent=None):
        super().__init__()

        self.__parent = parent or ContextRoot()

    def __getattr__(self, name):
        return getattr(self.__parent, name)


class ParsingObject(object):

    class _Shared(object):

        def __init__(self, grammar, stream, cache):
            super().__init__()

            self.grammar = grammar
            self.stream = stream
            self.cache = {} if cache else None

            self.rmf = None  # rightmost failure exception
            self.catch_failures = True

    def __init__(self, shared, context=None, alias=None):
        super().__init__()

        self.shared = shared
        self.context = context or Context()
        self.alias = alias

        self.grammar = shared.grammar
        self.stream = shared.stream
        self.cache = shared.cache

        self.read = self.stream.read
        self.seek = self.stream.seek
        self.tell = self.stream.tell

        try:
            self.peek = self.stream.peek
        except AttributeError:
            pass

    @classmethod
    def root(cls, grammar, stream, cache=True):
        return cls(cls._Shared(grammar, stream, cache))

    def child(self):
        return self.__class__(self.shared, Context(self.context), self.alias)

    @property
    def rmf(self):
        return self.shared.rmf

    @rmf.setter
    def rmf(self, value):
        self.shared.rmf = value

    @property
    def catch_failures(self):
        return self.shared.catch_failures

    @catch_failures.setter
    def catch_failures(self, value):
        self.shared.catch_failures = value

    def failed(self, expected, found=None):
        if self.alias:
            expected = {self.alias, }

        elif isinstance(expected, (tuple, list, set)):
            expected = set(expected)

        else:
            expected = {expected, }

        position = self.tell()

        if found is None:
            found = self.peek(1) or {EOI, }

        exception = MatchFailed(position, expected, found)

        if self.catch_failures and expected != {EOI, }:
            rmf = self.rmf

            if rmf is None or rmf.position < position:
                self.rmf = exception

            elif rmf.position == position:
                rmf.expected.update(expected)

        return exception

    def get_rule(self, name):
        try:
            return getattr(self.grammar, name)
        except AttributeError:
            raise GrammarError("rule {} not found".format(quote(name)))

    def get_action(self, name):
        try:
            return getattr(self.grammar, "_act_{}".format(name))
        except AttributeError:
            raise GrammarError("action {} not found".format(quote(name)))

    def get_predicate(self, name):
        try:
            return getattr(self.grammar, "_pre_{}".format(name))
        except AttributeError:
            raise GrammarError("predicate {} not found".format(quote(name)))

    def peek(self, size=-1):
        pos = self.tell()
        data = self.read(size)
        self.seek(pos)
        return data


class ExpressionBase(object):

    priority = None

    def __init__(self):
        super().__init__()

    def __call__(self, po):
        raise NotImplementedError()

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __eq__(self, other):
        return self.priority == other.priority

    def __ne__(self, other):
        return self.priority != other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority


class Reference(ExpressionBase):

    priority = 0

    def __init__(self, name):
        super().__init__()

        self.name = name

    def __call__(self, po):
        rule = po.get_rule(self.name)
        return rule(po.child())

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.name)


class Char(ExpressionBase):

    priority = 0

    def __init__(self):
        super().__init__()

    def __call__(self, po):
        char = po.read(1)

        if not char:
            raise po.failed(ANY, EOI)

        return char

    def __str__(self):
        return "."

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class MatchString(ExpressionBase):

    priority = Char.priority

    def __init__(self, pattern, ignore_case=False):
        super().__init__()

        self.pattern = pattern
        self.ignore_case = ignore_case

    def __call__(self, po):
        size = len(self.pattern)
        pos = po.tell()
        data = po.read(size)

        if len(data) < size:
            po.seek(pos)
            raise po.failed(str(self), EOI)

        pattern = self.pattern

        if self.ignore_case:
            pattern = pattern.lower()
            data = data.lower()

        if pattern != data:
            po.seek(pos)
            raise po.failed(str(self), data)

        return data

    def __str__(self):
        parts = [quote(self.pattern), ]

        if self.ignore_case:
            parts.append("i")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r}, {!r})".format(
            self.__class__.__name__, self.pattern, self.ignore_case)


class MatchClass(ExpressionBase):

    priority = Char.priority

    def __init__(self, pattern, ignore_case=False):
        super().__init__()

        self._pattern = pattern
        self._ignore_case = ignore_case

        self._regex = None
        self._update_regex()

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, value):
        self._pattern = value
        self._update_regex()

    @property
    def ignore_case(self):
        return self._ignore_case

    @ignore_case.setter
    def ignore_case(self, value):
        self._ignore_case = value
        self._update_regex()

    def _update_regex(self):
        flags = re.I if self.ignore_case else 0
        self._regex = re.compile("^[{}]$".format(self.pattern), flags)

    def __call__(self, po):
        pos = po.tell()
        char = po.read(1)

        if not char:
            raise po.failed(str(self), EOI)

        match = self._regex.match(char)

        if not match:
            po.seek(pos)
            raise po.failed(str(self), char)

        return match.group(0)

    def __str__(self):
        parts = ["[", self.pattern, "]"]

        if self.ignore_case:
            parts.append("i")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r}, {!r})".format(
            self.__class__.__name__, self.pattern, self.ignore_case)


class AndPredicate(ExpressionBase):

    priority = Char.priority

    def __init__(self, name):
        super().__init__()

        self.name = name

    def __call__(self, po):
        predicate = po.get_predicate(self.name)
        catch_failures = po.catch_failures
        po.catch_failures = False

        try:
            if not predicate(po.context):
                raise po.failed(EOI)
        finally:
            po.catch_failures = catch_failures

        return Nothing

    def __str__(self):
        return "".join(("&{", self.name, "}"))

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.name)


class NotPredicate(ExpressionBase):

    priority = Char.priority

    def __init__(self, name):
        super().__init__()

        self.name = name

    def __call__(self, po):
        predicate = po.get_predicate(self.name)
        catch_failures = po.catch_failures
        po.catch_failures = False

        try:
            if predicate(po.context):
                raise po.failed(EOI)
        finally:
            po.catch_failures = catch_failures

        return Nothing

    def __str__(self):
        return "".join(("!{", self.name, "}"))

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.name)


class Maybe(ExpressionBase):

    priority = Char.priority - 1

    def __init__(self, expression):
        super().__init__()

        self.expression = expression

    def __call__(self, po):
        try:
            return self.expression(po)
        except MatchFailed:
            return Nothing

    def __str__(self):
        need_brackets = self > self.expression
        parts = []

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        parts.append("?")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.expression)


class Some(ExpressionBase):

    priority = Maybe.priority

    def __init__(self, expression):
        super().__init__()

        self.expression = expression

    def __call__(self, po):
        result = [self.expression(po), ]

        while True:
            try:
                result.append(self.expression(po))
            except MatchFailed:
                return result

    def __str__(self):
        need_brackets = self > self.expression
        parts = []

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        parts.append("+")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.expression)


class Any(ExpressionBase):

    priority = Maybe.priority

    def __init__(self, expression):
        super().__init__()

        self.expression = expression

    def __call__(self, po):
        result = []

        while True:
            try:
                result.append(self.expression(po))
            except MatchFailed:
                return result or Nothing

    def __str__(self):
        need_brackets = self > self.expression
        parts = []

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        parts.append("*")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.expression)


class And(ExpressionBase):

    priority = Maybe.priority - 1

    def __init__(self, expression):
        super().__init__()

        self.expression = expression

    def __call__(self, po):
        catch_failures = po.catch_failures
        po.catch_failures = False
        pos = po.tell()

        try:
            self.expression(po)
        finally:
            po.catch_failures = catch_failures
            po.seek(pos)

        return Nothing

    def __str__(self):
        need_brackets = self > self.expression
        parts = ["&", ]

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.expression)


class Not(ExpressionBase):

    priority = And.priority

    def __init__(self, expression):
        super().__init__()

        self.expression = expression

    def __call__(self, po):
        catch_failures = po.catch_failures
        po.catch_failures = False
        pos = po.tell()

        try:
            self.expression(po)
        except MatchFailed:
            return Nothing
        finally:
            po.catch_failures = catch_failures
            po.seek(pos)

        raise po.failed(EOI)

    def __str__(self):
        need_brackets = self > self.expression
        parts = ["!", ]

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.expression)


def text(obj):
    def to_str(obj):
        if isinstance(obj, list):
            return "".join(to_str(x) for x in obj)

        return str(obj)

    return to_str(obj)


class Text(ExpressionBase):

    priority = And.priority

    def __init__(self, expression):
        super().__init__()

        self.expression = expression

    def __call__(self, po):
        return text(self.expression(po))

    def __str__(self):
        need_brackets = self > self.expression
        parts = ["$", ]

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.expression)


class Label(ExpressionBase):

    priority = Text.priority

    def __init__(self, label, expression):
        super().__init__()

        self.label = label
        self.expression = expression

    def __call__(self, po):
        result = self.expression(po)
        setattr(po.context, self.label, result)
        return result

    def __str__(self):
        need_brackets = self > self.expression
        parts = [self.label, ":"]

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r}, {!r})".format(
            self.__class__.__name__, self.label, self.expression)


class Sequence(ExpressionBase):

    priority = Label.priority - 1

    def __init__(self, expression, *expressions):
        super().__init__()

        self.expressions = (expression, ) + expressions

    def __call__(self, po):
        pos = po.tell()

        try:
            return [expression(po) for expression in self.expressions]
        except MatchFailed:
            po.seek(pos)
            raise

    def as_list(self):
        lst = []

        for expression in self.expressions:
            need_brackets = self >= expression
            parts = []

            if need_brackets:
                parts.append("(")

            parts.append(str(expression))

            if need_brackets:
                parts.append(")")

            lst.append("".join(parts))

        return lst

    def __str__(self):
        return " ".join(self.as_list())

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(repr(expr) for expr in self.expressions))


def Repeat(expression, times):
    args = (expression, ) * times
    return Sequence(*args)


class Action(ExpressionBase):

    priority = Sequence.priority - 1

    def __init__(self, name, expression):
        super().__init__()

        self.name = name
        self.expression = expression

    def __call__(self, po):
        result = self.expression(po)

        if self.name is None:
            return result

        function = po.get_action(self.name)
        return function(po.context)

    def as_list(self):
        if not isinstance(self.expression, Sequence):
            return [str(self), ]

        lst = self.expression.as_list()

        if self.name is not None:
            lst.append("{" + self.name + "}")
        else:
            lst.append("{}")

        return lst

    def __str__(self):
        need_brackets = self > self.expression
        parts = []

        if need_brackets:
            parts.append("(")

        parts.append(str(self.expression))

        if need_brackets:
            parts.append(")")

        parts.append(" {")

        if self.name is not None:
            parts.append(self.name)

        parts.append("}")

        return "".join(parts)

    def __repr__(self):
        return "{}({!r}, {!r})".format(
            self.__class__.__name__, self.name, self.expression)


class Choice(ExpressionBase):

    priority = Action.priority - 1

    def __init__(self, expression, *expressions):
        super().__init__()

        self.expressions = (expression, ) + expressions

    def __call__(self, po):
        pos = po.tell()
        last_failure = None

        for expression in self.expressions:
            try:
                return expression(po)
            except MatchFailed as e:
                po.seek(pos)
                last_failure = e

        raise last_failure or po.failed(EOI)

    def as_list(self):
        lst = []

        for expression in self.expressions:
            parts = []
            need_brackets = self >= expression

            if need_brackets:
                parts.append("(")

            parts.append(str(expression))

            if need_brackets:
                parts.append(")")

            lst.append("".join(parts))

        return lst

    def __str__(self):
        return " / ".join(self.as_list())

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(repr(expr) for expr in self.expressions))


class Rule(object):

    def __init__(self, name, expression, alias=None):
        super().__init__()

        self.name = name
        self.expression = expression
        self.alias = alias

    def __call__(self, po):
        if self.alias:
            po.alias = Marker(self.alias)

        if po.cache is None:
            return self.expression(po)

        key = (self.name, po.tell())

        try:
            pos, result = po.cache[key]
        except KeyError:
            pass
        else:
            if pos is None:
                raise result

            po.seek(pos)
            return result

        try:
            result = self.expression(po)
        except MatchFailed as e:
            po.cache[key] = (None, e)
            raise
        else:
            po.cache[key] = (po.tell(), result)
            return result

    def __str__(self):
        parts = [self.name, ]

        if self.alias:
            parts.append(" ")
            parts.append(quote(self.alias))

        parts.append("\n")
        parts.append("  = ")

        if isinstance(self.expression, Choice):
            lst = self.expression.as_list()
            parts.append(lst[0])

            for x in lst[1:]:
                parts.append("\n  / ")
                parts.append(x)

        elif isinstance(self.expression, (Action, Sequence)):
            lst = self.expression.as_list()
            lst80 = []
            line = lst[0]

            for x in lst[1:]:
                if len(line) + len(x) <= 75:
                    line = "{} {}".format(line, x)

                else:
                    lst80.append(line)
                    line = x

            lst80.append(line)
            parts.append(lst80[0])

            for x in lst80[1:]:
                parts.append("\n    ")
                parts.append(x)

        else:
            parts.append(str(self.expression))

        return "".join(parts)

    def __repr__(self):
        args = [repr(self.name), repr(self.expression)]

        if self.alias is not None:
            args.append(repr(self.alias))

        return "{}({})".format(self.__class__.__name__, ", ".join(args))


class As(object):

    def __init__(self, name, expression):
        super().__init__()

        self.name = name
        self.expression = expression


class GrammarMeta(type):

    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return OrderedDict()

    def __init__(cls, name, bases, namespace, **kwds):
        super().__init__(name, bases, namespace)

        cls.__order__ = []

        for name, value in list(namespace.items()):
            if isinstance(value, (ExpressionBase, As)):
                setattr(cls, name, value)

        try:
            grammar = getattr(cls, "__grammar__")
        except AttributeError:
            pass
        else:
            parse = PEGGrammar()

            for rule in parse(grammar):
                setattr(cls, rule.name, rule)

    def __setattr__(cls, name, value):
        if isinstance(value, ExpressionBase):
            value = Rule(name, value)
            cls.__order__.append(name)

        elif isinstance(value, As):
            value = Rule(name, value.expression, value.name)
            cls.__order__.append(name)

        elif isinstance(value, Rule):
            cls.__order__.append(name)

        super().__setattr__(name, value)

    def __delattr__(cls, name):
        super().__delattr__(name)

        try:
            cls.__order__.remove(name)
        except ValueError:
            pass


class Grammar(object, metaclass=GrammarMeta):

    __order__ = []

    def __init__(self, cache=True, line_column=True):
        super().__init__()

        self._cache = cache
        self._line_column = line_column

    def __call__(self, stream):
        if isinstance(stream, str):
            stream = io.StringIO(stream)

        po = ParsingObject.root(self, stream, self._cache)

        try:
            Start = getattr(self, "Start")
        except AttributeError:
            Start = getattr(self, self.__order__[0])

        try:
            result = Start(po)
        except MatchFailed:
            pass
        else:
            if not po.peek(1):
                return result

        exception = po.rmf or po.failed(EOI)

        if self._line_column:
            position = exception.position
            po.seek(0)
            content = po.read(position)
            line = content.count("\n") + 1
            column = position - (content.rfind("\n") if line > 1 else 0) + 1
            exception.coordinates = (line, column)

        raise exception

    def __str__(self):
        return "\n\n".join(str(getattr(self, name)) for name in self.__order__)

    def __repr__(self):
        return "{}(cache={!r}, line_column={!r})".format(
            self.__class__.__name__, self._cache, self._line_column)


def _token(name):
    return Sequence(MatchString(name), Not(Reference("IdentifierPart")))


class PEGGrammar(Grammar):

    def __init__(self, cache=True, line_column=True, encoding=None):
        super().__init__(cache, line_column)

        self._encoding = encoding or getpreferredencoding(False)

    #####################
    # Syntactic Grammar #
    #####################

    Grammar = Action("grammar", Sequence(
        Reference("__"),
        Label("rules", Some(Sequence(
            Reference("Rule"),
            Reference("__"),
        )))
    ))

    @staticmethod
    def _act_grammar(ctx):
        return [rule[0] for rule in ctx.rules]

    Rule = Action("rule", Sequence(
        Label("name", Reference("Identifier")),
        Reference("__"),
        Label("alias", Maybe(Sequence(
            Reference("StringLiteral"),
            Reference("__"),
        ))),
        MatchString("="), Reference("__"),
        Label("expression", Reference("Expression")),
        Reference("EOS"),
    ))

    @staticmethod
    def _act_rule(ctx):
        alias = ctx.alias[0] if ctx.alias else None
        return Rule(ctx.name, ctx.expression, alias)

    Expression = Reference("ChoiceExpression")

    ChoiceExpression = Action("expression_choice", Sequence(
        Label("first", Reference("ActionExpression")),
        Label("rest", Any(Sequence(
            Reference("__"), MatchString("/"), Reference("__"),
            Reference("ActionExpression"),
        )))
    ))

    @staticmethod
    def _act_expression_choice(ctx):
        if not ctx.rest:
            return ctx.first

        expressions = (ctx.first, ) + tuple(elem[3] for elem in ctx.rest)
        return Choice(*expressions)

    ActionExpression = Action("expression_action", Sequence(
        Label("expression", Reference("SequenceExpression")),
        Label("name", Maybe(Sequence(
            Reference("__"),
            Reference("CodeReference"),
        )))
    ))

    @staticmethod
    def _act_expression_action(ctx):
        if not ctx.name:
            return ctx.expression

        return Action(ctx.name[1], ctx.expression)

    SequenceExpression = Action("expression_sequence", Sequence(
        Label("first", Reference("LabeledExpression")),
        Label("rest", Any(Sequence(
            Reference("__"),
            Reference("LabeledExpression"),
        )))
    ))

    @staticmethod
    def _act_expression_sequence(ctx):
        if not ctx.rest:
            return ctx.first

        expressions = (ctx.first, ) + tuple(elem[1] for elem in ctx.rest)
        return Sequence(*expressions)

    LabeledExpression = Choice(
        Action("expression_labeled", Sequence(
            Label("name", Reference("Identifier")),
            Reference("__"), MatchString(":"), Reference("__"),
            Label("expression", Reference("PrefixedExpression")),
        )),
        Reference("PrefixedExpression"),
    )

    @staticmethod
    def _act_expression_labeled(ctx):
        return Label(ctx.name, ctx.expression)

    PrefixedExpression = Choice(
        Action("expression_prefixed", Sequence(
            Label("operator", Reference("PrefixedOperator")),
            Reference("__"),
            Label("expression", Reference("SuffixedExpression")),
        )),
        Reference("SuffixedExpression"),
    )

    PrefixedOperator = Choice(
        MatchString("$"),
        MatchString("&"),
        MatchString("!"),
    )

    @staticmethod
    def _act_expression_prefixed(ctx):
        cls = {"$": Text, "&": And, "!": Not}[ctx.operator]
        return cls(ctx.expression)

    SuffixedExpression = Choice(
        Action("expression_suffixed", Sequence(
            Label("expression", Reference("PrimaryExpression")),
            Reference("__"),
            Label("operator", Reference("SuffixedOperator")),
        )),
        Reference("PrimaryExpression"),
    )

    SuffixedOperator = Choice(
        MatchString("?"),
        MatchString("*"),
        MatchString("+"),
    )

    @staticmethod
    def _act_expression_suffixed(ctx):
        cls = {"?": Maybe, "*": Any, "+": Some}[ctx.operator]
        return cls(ctx.expression)

    PrimaryExpression = Choice(
        Reference("LiteralMatcher"),
        Reference("CharacterClassMatcher"),
        Reference("AnyMatcher"),
        Reference("RuleReferenceExpression"),
        Reference("SemanticPredicateExpression"),
        Action("expression", Sequence(
            MatchString("("), Reference("__"),
            Label("expression", Reference("Expression")),
            Reference("__"), MatchString(")"),
        ))
    )

    @staticmethod
    def _act_expression(ctx):
        return ctx.expression

    RuleReferenceExpression = Action("expression_reference", Sequence(
        Label("name", Reference("Identifier")),
        Not(Sequence(
            Reference("__"),
            Maybe(Sequence(
                Reference("StringLiteral"),
                Reference("__"),
            )),
            MatchString("="),
        ))
    ))

    @staticmethod
    def _act_expression_reference(ctx):
        return Reference(ctx.name)

    SemanticPredicateExpression = Action("expression_predicate", Sequence(
        Label("operator", Reference("SemanticPredicateOperator")),
        Reference("__"),
        Label("name", Reference("CodeReference")),
    ))

    SemanticPredicateOperator = Choice(
        MatchString("&"),
        MatchString("!"),
    )

    @staticmethod
    def _act_expression_predicate(ctx):
        cls = {"&": AndPredicate, "!": NotPredicate}[ctx.operator]
        return cls(ctx.name)

    ###################
    # Lexical Grammar #
    ###################

    WhiteSpace = As("whitespace", Choice(
        MatchString("\t"),
        MatchString("\v"),
        MatchString("\f"),
        MatchString(" "),
    ))

    LineTerminator = Choice(
        MatchString("\n"),
        MatchString("\r"),
    )

    LineTerminatorSequence = As("end of line", Choice(
        MatchString("\n"),
        MatchString("\r\n"),
        MatchString("\r"),
    ))

    Comment = As("comment", Sequence(
        MatchString("#"),
        Any(Sequence(
            Not(Reference("LineTerminator")),
            Char(),
        )),
    ))

    Identifier = Text(Sequence(
        Not(Reference("ReservedWord")),
        Reference("IdentifierName"),
    ))

    IdentifierName = As("identifier", Choice(
        Sequence(
            Reference("IdentifierStart"),
            Any(Reference("IdentifierPart")),
        ),
        Some(Reference("Underscore")),
    ))

    IdentifierStart = Reference("Alphabetic")

    IdentifierPart = Choice(
        Reference("IdentifierStart"),
        Reference("DecimalDigit"),
        Reference("Underscore"),
    )

    ReservedWord = Choice(*(Reference(name) for name in (
        "FalseToken", "NoneToken", "TrueToken", "AndToken", "AsToken",
        "AssertToken", "BreakToken", "ClassToken", "ContinueToken", "DefToken",
        "DelToken", "ElifToken", "ElseToken", "ExceptToken", "FinallyToken",
        "ForToken", "FromToken", "GlobalToken", "IfToken", "ImportToken",
        "InToken", "IsToken", "LambdaToken", "NonlocalToken", "NotToken",
        "OrToken", "PassToken", "RaiseToken", "ReturnToken", "TryToken",
        "WhileToken", "WithToken", "YieldToken",
    )))

    LiteralMatcher = As("literal", Action("matcher_literal", Sequence(
        Label("pattern", Reference("StringLiteral")),
        Label("ignore_case", Maybe(MatchString("i"))),
    )))

    @staticmethod
    def _act_matcher_literal(ctx):
        return MatchString(ctx.pattern, bool(ctx.ignore_case))

    StringLiteral = As("string", Choice(
        Action("unescape_string", Sequence(
            MatchString('"'),
            Label("value", Text(Any(Reference("DoubleQuoteStringCharacter")))),
            MatchString('"'),
        )),
        Action("unescape_string", Sequence(
            MatchString("'"),
            Label("value", Text(Any(Reference("SingleQuoteStringCharacter")))),
            MatchString("'"),
        )),
    ))

    def _act_unescape_string(self, ctx):
        unescape_map = {
            '\\"': '"',  "\\'": "'",  "\\\\": "\\", "\\a": "\a", "\\b": "\b",
            "\\t": "\t", "\\n": "\n", "\\v":  "\v", "\\f": "\f", "\\r": "\r",
        }
        s = "".join(unescape_map.get(x, x) for x in
                    re.split(r"(\\[ux\"'\\abtnvfrn])", ctx.value))

        if "\\u" in s:
            s = re.sub(r"\\u([0-9A-Za-z]{4})",
                       lambda m: chr(int(m.group(1), 16)), s)

        if "\\x" in s:
            s = re.sub(br"\\x([0-9A-Za-z]{2})",
                       lambda m: bytes((int(m.group(1), 16), )),
                       s.encode(self._encoding)).decode(self._encoding)

        return s

    DoubleQuoteStringCharacter = Choice(
        Sequence(
            Not(Choice(
                MatchString('"'),
                MatchString("\\"),
                Reference("LineTerminator"),
            )),
            Char(),
        ),
        Sequence(
            MatchString("\\"),
            Not(Reference("LineTerminator")),
            Char(),
        ),
        Reference("LineContinuation"),
    )

    SingleQuoteStringCharacter = Choice(
        Sequence(
            Not(Choice(
                MatchString("'"),
                MatchString("\\"),
                Reference("LineTerminator"),
            )),
            Char(),
        ),
        Sequence(
            MatchString("\\"),
            Not(Reference("LineTerminator")),
            Char(),
        ),
        Reference("LineContinuation"),
    )

    CharacterClassMatcher = As(
        "character class",
        Action("matcher_class", Sequence(
            MatchString("["),
            Label("pattern", Text(Any(Reference("ClassCharacter")))),
            MatchString("]"),
            Label("ignore_case", Maybe(MatchString("i"))),
        ))
    )

    @staticmethod
    def _act_matcher_class(ctx):
        return MatchClass(ctx.pattern, bool(ctx.ignore_case))

    ClassCharacter = Choice(
        Sequence(
            Not(
                Choice(
                    MatchString("]"),
                    MatchString("\\"),
                    Reference("LineTerminator"),
                ),
            ),
            Char(),
        ),
        Sequence(
            MatchString("\\"),
            Not(Reference("LineTerminator")),
            Char(),
        ),
        Reference("LineContinuation"),
    )

    LineContinuation = Action("line_continuation", Sequence(
        MatchString("\\"),
        Reference("LineTerminatorSequence"),
    ))

    @staticmethod
    def _act_line_continuation(ctx):
        return ""

    DecimalDigit = MatchClass("0-9")

    Alphabetic = MatchClass("a-z", ignore_case=True)

    Underscore = MatchString("_")

    AnyMatcher = Action("matcher_any", MatchString("."))

    @staticmethod
    def _act_matcher_any(ctx):
        return Char()

    CodeReference = As(
        "code reference",
        Action("code_reference", Sequence(
            MatchString("{"),
            Label("name", Maybe(Reference("Identifier"))),
            MatchString("}"),
        ))
    )

    @staticmethod
    def _act_code_reference(ctx):
        return ctx.name or None

    # Tokens

    FalseToken    = _token("False")
    NoneToken     = _token("None")
    TrueToken     = _token("True")
    AndToken      = _token("and")
    AsToken       = _token("as")
    AssertToken   = _token("assert")
    BreakToken    = _token("break")
    ClassToken    = _token("class")
    ContinueToken = _token("continue")
    DefToken      = _token("def")
    DelToken      = _token("del")
    ElifToken     = _token("elif")
    ElseToken     = _token("else")
    ExceptToken   = _token("except")
    FinallyToken  = _token("finally")
    ForToken      = _token("for")
    FromToken     = _token("from")
    GlobalToken   = _token("global")
    IfToken       = _token("if")
    ImportToken   = _token("import")
    InToken       = _token("in")
    IsToken       = _token("is")
    LambdaToken   = _token("lambda")
    NonlocalToken = _token("nonlocal")
    NotToken      = _token("not")
    OrToken       = _token("or")
    PassToken     = _token("pass")
    RaiseToken    = _token("raise")
    ReturnToken   = _token("return")
    TryToken      = _token("try")
    WhileToken    = _token("while")
    WithToken     = _token("with")
    YieldToken    = _token("yield")

    # Skipped

    __ = Any(Choice(
        Reference("WhiteSpace"),
        Reference("LineTerminatorSequence"),
        Reference("Comment"),
    ))

    _ = Any(Reference("WhiteSpace"))

    # Automatic Semicolon Insertion

    EOS = Choice(
        Sequence(Reference("__"), MatchString(";")),
        Sequence(
            Reference("_"),
            Maybe(Reference("Comment")),
            Reference("LineTerminatorSequence"),
        ),
        Sequence(Reference("__"), Reference("EOF")),
    )

    EOF = Not(Char())


if __name__ == "__main__":
    pass
    # print(PEGGrammar()(open("PEG.txt").read()))
    # print(PEGGrammar())
    # print(str(PEGGrammar()(str(PEGGrammar()))))
