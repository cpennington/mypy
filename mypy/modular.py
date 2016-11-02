import sys
from mypy.parse import parse
from mypy.options import Options
from mypy.traverser import TraverserVisitor

from collections import namedtuple, defaultdict

import mypy.nodes
from mypy.nodes import IntExpr, Node, MypyFile, SymbolTable
from mypy.build import find_module, default_lib_path, default_data_dir

from typing import Iterable, Union, Optional


class ModuleNotFound(Exception):
    def __init__(self, module):
        self.module = module
        super().__init__("Module {} not found".format(module))


class NameNotFound(Exception):
    def __init__(self, module, name):
        self.module = module
        self.name = name
        super().__init__("{} not found in module {}".format(name, module))


class SubmoduleNotFound(NameNotFound): pass


class SignalDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signals = defaultdict(list)

    def on_change(self, key, func, *args, **kwargs):
        self._signals[key].append((func, args, kwargs))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        if key in self._signals:
            for (func, args, kwargs) in self._signals.pop(key):
                func(*args, **kwargs)



class SignallingSymbolTable(SignalDict, SymbolTable):
    pass


class Environment:
    def __init__(self, parent: Optional['Environment'] = None) -> None:

        self.parent = parent
        if parent:
            self.modules = parent.modules
        else:
            self.modules = SignalDict()

        self.names = {}

    def named_type(self, module, name, args=None):
        return self.lookup_fully_qualified(module, name)

    def lookup_fully_qualified(self, module: str, name: str) -> Node:
        """Lookup a qualified name in this environment.

        Assume that the name is defined. This happens in the global namespace -- the local
        module namespace is ignored.
        """
        if module not in self.modules:
            raise ModuleNotFound(module)

        if name not in self.modules[module].names:
            raise NameNotFound(self.modules[module], name)

        return self.module[module].names[name]


class RuleResult:
    pass

class RetryResult(RuleResult):
    def __init__(self, rule, env, expr):
        self.rule = rule
        self.env = env
        self.expr = expr


class Infer(RuleResult):
    def __init__(self, iexpr, itype):
        self.itype = itype
        self.iexpr = iexpr
        super().__init__()

    def execute(self, engine):
        engine.set_type(self.iexpr, self.itype)


class RetryOnChange(RetryResult):
    def __init__(self, rule, env, expr, in_env, name):
        self.in_env = in_env
        self.name = name
        super().__init__(rule, env, expr)

    def execute(self, engine):
        self.in_env.on_change(
            self.name,
            engine.schedule_rule,
            self.rule,
            self.expr
        )


class LoadModule(RetryResult):
    def __init__(self, rule, env, expr, in_env, module):
        self.in_env = in_env
        self.module = module
        super().__init__(rule, env, expr)

    def execute(self, engine):
        if self.module in self.in_env.modules:
            return

        self.in_env.modules.on_change(
            self.module,
            engine.schedule_rule,
            self.rule,
            self.expr
        )

        file_id = self.module
        if engine.options.python_version[0] == 2:
            # The __builtin__ module is called internally by mypy
            # 'builtins' in Python 2 mode (similar to Python 3),
            # but the stub file is __builtin__.pyi.  The reason is
            # that a lot of code hard-codes 'builtins.x' and it's
            # easier to work it around like this.  It also means
            # that the implementation can mostly ignore the
            # difference and just assume 'builtins' everywhere,
            # which simplifies code.
            file_id = '__builtin__'
        else:
            file_id = 'builtins'
        data_dir = default_data_dir(None)
        lib_path = default_lib_path(data_dir, engine.options.python_version)
        path = find_module(file_id, lib_path)
        self.in_env.modules[self.module] = engine.load_module(path)


class IntLiteral:
    applies_to = IntExpr

    def apply(self, env: Environment, expr: IntExpr) -> Iterable[RuleResult]:
        try:
            env.named_type('builtins', 'int')
        except ModuleNotFound as exc:
            yield LoadModule(self, env, expr, env, exc.module)
            return

        except SubmoduleNotFound as exc:
            yield LoadModule(self, env, expr, env, "{}.{}".join(exc.module, exc.name))
            return

        except NameNotFound as exc:
            yield RetryOnChange(self, env, expr, exc.module.names, exc.name)
            return

        yield Infer(expr, env.named_type('builtins.int'))


class InsertBuiltins:
    applies_to = MypyFile

    def apply(self, env: Environment, expr: MypyFile) -> Iterable[RuleResult]:
        if 'builtins' not in env.modules:
            yield LoadModule(self, env, expr, env, 'builtins')
            return

        builtin_module = env.modules['builtins']

        for name, type in env.environment_for(builtin_module).names.items():
            env.bind(name, type)


RULES = [
    IntLiteral(),
    InsertBuiltins(),
]


class RuleVisitor(TraverserVisitor):

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.current_scope = None

    # Module structure

    def visit_mypy_file(self, o: mypy.nodes.MypyFile) -> None:
        with self.set_scope(Environment(self.engine.builtins_environment)):
            self.engine.env_for_expr[o] = self.current_scope
            self.engine.run_rules(o)
            super().visit_mypy_file(o)

    def visit_import(self, o: mypy.nodes.Import) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_import(o)

    def visit_import_from(self, o: mypy.nodes.ImportFrom) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_import_from(o)

    def visit_import_all(self, o: mypy.nodes.ImportAll) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_import_all(o)

    # Definitions

    def visit_func_def(self, o: mypy.nodes.FuncDef) -> None:
        with self.set_scope(Environment(self.current_scope)):
            self.engine.env_for_expr[o] = self.current_scope
            self.engine.run_rules(o)
            super().visit_func_def(o)

    def visit_overloaded_func_def(self,
                                  o: mypy.nodes.OverloadedFuncDef) -> None:
        with self.set_scope(Environment(self.current_scope)):
            self.engine.env_for_expr[o] = self.current_scope
            self.engine.run_rules(o)
            super().visit_overloaded_func_def(o)

    def visit_class_def(self, o: mypy.nodes.ClassDef) -> None:
        with self.set_scope(ClassScope(self.current_scope)):
            self.engine.env_for_expr[o] = self.current_scope
            self.engine.run_rules(o)
            super().visit_class_def(o)

    def visit_global_decl(self, o: mypy.nodes.GlobalDecl) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_global_decl(o)

    def visit_nonlocal_decl(self, o: mypy.nodes.NonlocalDecl) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_nonlocal_decl(o)

    def visit_decorator(self, o: mypy.nodes.Decorator) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_decorator(o)

    def visit_var(self, o: mypy.nodes.Var) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_var(o)

    # Statements

    def visit_block(self, o: mypy.nodes.Block) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_block(o)

    def visit_expression_stmt(self, o: mypy.nodes.ExpressionStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_expression_stmt(o)

    def visit_assignment_stmt(self, o: mypy.nodes.AssignmentStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_assignment_stmt(o)

    def visit_operator_assignment_stmt(self,
                                       o: mypy.nodes.OperatorAssignmentStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_operator_assignment_stmt(o)

    def visit_while_stmt(self, o: mypy.nodes.WhileStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_while_stmt(o)

    def visit_for_stmt(self, o: mypy.nodes.ForStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_for_stmt(o)

    def visit_return_stmt(self, o: mypy.nodes.ReturnStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_return_stmt(o)

    def visit_assert_stmt(self, o: mypy.nodes.AssertStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_assert_stmt(o)

    def visit_del_stmt(self, o: mypy.nodes.DelStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_del_stmt(o)

    def visit_if_stmt(self, o: mypy.nodes.IfStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_if_stmt(o)

    def visit_break_stmt(self, o: mypy.nodes.BreakStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_break_stmt(o)

    def visit_continue_stmt(self, o: mypy.nodes.ContinueStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_continue_stmt(o)

    def visit_pass_stmt(self, o: mypy.nodes.PassStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_pass_stmt(o)

    def visit_raise_stmt(self, o: mypy.nodes.RaiseStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_raise_stmt(o)

    def visit_try_stmt(self, o: mypy.nodes.TryStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_try_stmt(o)

    def visit_with_stmt(self, o: mypy.nodes.WithStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_with_stmt(o)

    def visit_print_stmt(self, o: mypy.nodes.PrintStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_print_stmt(o)

    def visit_exec_stmt(self, o: mypy.nodes.ExecStmt) -> None:
        self.engine.env_for_expr[o] = self.current_scope
        self.engine.run_rules(o)
        super().visit_exec_stmt(o)

    # Expressions

    def visit_int_expr(self, o: mypy.nodes.IntExpr) -> None:
        self.engine.run_rules(o)
        super().visit_int_expr(o)

    def visit_str_expr(self, o: mypy.nodes.StrExpr) -> None:
        self.engine.run_rules(o)
        super().visit_str_expr(o)

    def visit_bytes_expr(self, o: mypy.nodes.BytesExpr) -> None:
        self.engine.run_rules(o)
        super().visit_bytes_expr(o)

    def visit_unicode_expr(self, o: mypy.nodes.UnicodeExpr) -> None:
        self.engine.run_rules(o)
        super().visit_unicode_expr(o)

    def visit_float_expr(self, o: mypy.nodes.FloatExpr) -> None:
        self.engine.run_rules(o)
        super().visit_float_expr(o)

    def visit_complex_expr(self, o: mypy.nodes.ComplexExpr) -> None:
        self.engine.run_rules(o)
        super().visit_complex_expr(o)

    def visit_ellipsis(self, o: mypy.nodes.EllipsisExpr) -> None:
        self.engine.run_rules(o)
        super().visit_ellipsis(o)

    def visit_star_expr(self, o: mypy.nodes.StarExpr) -> None:
        self.engine.run_rules(o)
        super().visit_star_expr(o)

    def visit_name_expr(self, o: mypy.nodes.NameExpr) -> None:
        self.engine.run_rules(o)
        super().visit_name_expr(o)

    def visit_member_expr(self, o: mypy.nodes.MemberExpr) -> None:
        self.engine.run_rules(o)
        super().visit_member_expr(o)

    def visit_yield_from_expr(self, o: mypy.nodes.YieldFromExpr) -> None:
        self.engine.run_rules(o)
        super().visit_yield_from_expr(o)

    def visit_yield_expr(self, o: mypy.nodes.YieldExpr) -> None:
        self.engine.run_rules(o)
        super().visit_yield_expr(o)

    def visit_call_expr(self, o: mypy.nodes.CallExpr) -> None:
        self.engine.run_rules(o)
        super().visit_call_expr(o)

    def visit_op_expr(self, o: mypy.nodes.OpExpr) -> None:
        self.engine.run_rules(o)
        super().visit_op_expr(o)

    def visit_comparison_expr(self, o: mypy.nodes.ComparisonExpr) -> None:
        self.engine.run_rules(o)
        super().visit_comparison_expr(o)

    def visit_cast_expr(self, o: mypy.nodes.CastExpr) -> None:
        self.engine.run_rules(o)
        super().visit_cast_expr(o)

    def visit_reveal_type_expr(self, o: mypy.nodes.RevealTypeExpr) -> None:
        self.engine.run_rules(o)
        super().visit_reveal_type_expr(o)

    def visit_super_expr(self, o: mypy.nodes.SuperExpr) -> None:
        self.engine.run_rules(o)
        super().visit_super_expr(o)

    def visit_unary_expr(self, o: mypy.nodes.UnaryExpr) -> None:
        self.engine.run_rules(o)
        super().visit_unary_expr(o)

    def visit_list_expr(self, o: mypy.nodes.ListExpr) -> None:
        self.engine.run_rules(o)
        super().visit_list_expr(o)

    def visit_dict_expr(self, o: mypy.nodes.DictExpr) -> None:
        self.engine.run_rules(o)
        super().visit_dict_expr(o)

    def visit_tuple_expr(self, o: mypy.nodes.TupleExpr) -> None:
        self.engine.run_rules(o)
        super().visit_tuple_expr(o)

    def visit_set_expr(self, o: mypy.nodes.SetExpr) -> None:
        self.engine.run_rules(o)
        super().visit_set_expr(o)

    def visit_index_expr(self, o: mypy.nodes.IndexExpr) -> None:
        self.engine.run_rules(o)
        super().visit_index_expr(o)

    def visit_type_application(self, o: mypy.nodes.TypeApplication) -> None:
        self.engine.run_rules(o)
        super().visit_type_application(o)

    def visit_func_expr(self, o: mypy.nodes.FuncExpr) -> None:
        self.engine.run_rules(o)
        super().visit_func_expr(o)

    def visit_list_comprehension(self, o: mypy.nodes.ListComprehension) -> None:
        self.engine.run_rules(o)
        super().visit_list_comprehension(o)

    def visit_set_comprehension(self, o: mypy.nodes.SetComprehension) -> None:
        self.engine.run_rules(o)
        super().visit_set_comprehension(o)

    def visit_dictionary_comprehension(self, o: mypy.nodes.DictionaryComprehension) -> None:
        self.engine.run_rules(o)
        super().visit_dictionary_comprehension(o)

    def visit_generator_expr(self, o: mypy.nodes.GeneratorExpr) -> None:
        self.engine.run_rules(o)
        super().visit_generator_expr(o)

    def visit_slice_expr(self, o: mypy.nodes.SliceExpr) -> None:
        self.engine.run_rules(o)
        super().visit_slice_expr(o)

    def visit_conditional_expr(self, o: mypy.nodes.ConditionalExpr) -> None:
        self.engine.run_rules(o)
        super().visit_conditional_expr(o)

    def visit_backquote_expr(self, o: mypy.nodes.BackquoteExpr) -> None:
        self.engine.run_rules(o)
        super().visit_backquote_expr(o)

    def visit_type_var_expr(self, o: mypy.nodes.TypeVarExpr) -> None:
        self.engine.run_rules(o)
        super().visit_type_var_expr(o)

    def visit_type_alias_expr(self, o: mypy.nodes.TypeAliasExpr) -> None:
        self.engine.run_rules(o)
        super().visit_type_alias_expr(o)

    def visit_namedtuple_expr(self, o: mypy.nodes.NamedTupleExpr) -> None:
        self.engine.run_rules(o)
        super().visit_namedtuple_expr(o)

    def visit_typeddict_expr(self, o: mypy.nodes.TypedDictExpr) -> None:
        self.engine.run_rules(o)
        super().visit_typeddict_expr(o)

    def visit_newtype_expr(self, o: mypy.nodes.NewTypeExpr) -> None:
        self.engine.run_rules(o)
        super().visit_newtype_expr(o)

    def visit__promote_expr(self, o: mypy.nodes.PromoteExpr) -> None:
        self.engine.run_rules(o)
        super().visit__promote_expr(o)

    def visit_await_expr(self, o: mypy.nodes.AwaitExpr) -> None:
        self.engine.run_rules(o)
        super().visit_await_expr(o)

    def visit_temp_node(self, o: mypy.nodes.TempNode) -> None:
        self.engine.run_rules(o)
        super().visit_temp_node(o)


class RulesEngine:
    def __init__(self, options):
        self.options = options
        self.result_queue = []
        self._rules_cache = {}
        self.builtins_environment = Environment()
        self._environs = defaultdict(lambda: self._root_environment)
        self._pending_rules = []
        self._types = {}

    def env_for_expr(self, expr):
        return self._environs[expr]

    def set_type(self, expr, type):
        self._types[expr] = type

    def schedule_rule(self, rule, expr):
        self._pending_rules.append((rule, expr))

    def run_rules(self, expr: mypy.nodes.Node):
        expr_type = type(expr)
        if expr_type not in self._rules_cache:
            self._rules_cache[expr_type] = [rule for rule in RULES if isinstance(expr, rule.applies_to)]

        for rule in self._rules_cache[expr_type]:
            self.run_rule(rule, expr)

    def run_rule(self, rule, expr):
        for result in rule.apply(self.env_for_expr(expr), expr):
            self.result_queue.append(result)

    def load_module(self, path):
        with open(path) as input_file:
            ast = parse(input_file.read(), path, None, self.options)
        ast.names = SignallingSymbolTable()
        ast.accept(RuleVisitor(self))
        return ast

    def resolve(self):
        while self.result_queue or self._pending_rules:
            while self.result_queue:
                next_result = self.result_queue.pop(0)
                next_result.execute(self)

            while self._pending_rules:
                rule, expr = self._pending_rules.pop(0)
                self.run_rule(rule, expr)


def main(args):
    filename = args[1]
    engine = RulesEngine(Options())
    engine.load_module(filename)
    engine.resolve()
    print(engine._types)
    print(engine._pending_rules)
    print(engine.result_queue)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
