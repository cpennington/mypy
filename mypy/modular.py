import sys
from mypy.parse import parse
from mypy.options import Options
from mypy.traverser import TraverserVisitor

from collections import namedtuple

import mypy.nodes
from mypy.nodes import IntExpr, Node

from typing import Iterable, Union, Optional

class Environment:
    def __init__(self, parent: Optional['Environment'] = None) -> None:

        self.parent = parent
        if parent:
            self.modules = parent.modules
        else:
            self.modules = {}

        self.names = {}

    def named_type(self, string, args=None):
        return self.lookup_fully_qualified(string)

    def lookup_fully_qualified(self, name: str) -> Node:
        """Lookup a qualified name in this environment.

        Assume that the name is defined. This happens in the global namespace -- the local
        module namespace is ignored.
        """
        assert '.' in name
        parts = name.split('.')
        n = self.modules[parts[0]]
        for i in range(1, len(parts) - 1):
            n = cast(MypyFile, n.names[parts[i]].node)
        return n.names[parts[-1]]


class BuiltinEnvironment(Environment):
    def __init__(self, options):
        super().__init__()
        if options.python_version[0] == 2:
            # The __builtin__ module is called internally by mypy
            # 'builtins' in Python 2 mode (similar to Python 3),
            # but the stub file is __builtin__.pyi.  The reason is
            # that a lot of code hard-codes 'builtins.x' and it's
            # easier to work it around like this.  It also means
            # that the implementation can mostly ignore the
            # difference and just assume 'builtins' everywhere,
            # which simplifies code.
            file_id = '__builtin__'
        path = find_module(file_id, manager.lib_path)

Infer = namedtuple('Infer', ['expr', 'type'])

RuleResult = Union[Infer]


class IntLiteral:
    applies_to = IntExpr

    def apply(self, env: Environment, expr: IntExpr) -> Iterable[RuleResult]:
        yield Infer(expr, env.named_type('builtins.int'))


RULES = [
    IntLiteral()
]


class RuleVisitor(TraverserVisitor):

    def __init__(self):
        super().__init__()
        self.root_environment = BuiltinEnvironment()
        self._rules_cache = {}

    def apply_rules(self, o: mypy.nodes.Node):
        type_o = type(o)
        if type_o not in self._rules_cache:
            self._rules_cache[type_o] = [rule for rule in RULES if isinstance(o, rule.applies_to)]

        for rule in self._rules_cache[type_o]:
            for result in rule.apply(self.root_environment, o):
                print(result)


    # Module structure

    def visit_mypy_file(self, o: mypy.nodes.MypyFile) -> None:
        self.apply_rules(o)
        super().visit_mypy_file(o)

    def visit_import(self, o: mypy.nodes.Import) -> None:
        self.apply_rules(o)
        super().visit_import(o)

    def visit_import_from(self, o: mypy.nodes.ImportFrom) -> None:
        self.apply_rules(o)
        super().visit_import_from(o)

    def visit_import_all(self, o: mypy.nodes.ImportAll) -> None:
        self.apply_rules(o)
        super().visit_import_all(o)

    # Definitions

    def visit_func_def(self, o: mypy.nodes.FuncDef) -> None:
        self.apply_rules(o)
        super().visit_func_def(o)

    def visit_overloaded_func_def(self,
                                  o: mypy.nodes.OverloadedFuncDef) -> None:
        self.apply_rules(o)
        super().visit_overloaded_func_def(o)

    def visit_class_def(self, o: mypy.nodes.ClassDef) -> None:
        self.apply_rules(o)
        super().visit_class_def(o)

    def visit_global_decl(self, o: mypy.nodes.GlobalDecl) -> None:
        self.apply_rules(o)
        super().visit_global_decl(o)

    def visit_nonlocal_decl(self, o: mypy.nodes.NonlocalDecl) -> None:
        self.apply_rules(o)
        super().visit_nonlocal_decl(o)

    def visit_decorator(self, o: mypy.nodes.Decorator) -> None:
        self.apply_rules(o)
        super().visit_decorator(o)

    def visit_var(self, o: mypy.nodes.Var) -> None:
        self.apply_rules(o)
        super().visit_var(o)

    # Statements

    def visit_block(self, o: mypy.nodes.Block) -> None:
        self.apply_rules(o)
        super().visit_block(o)

    def visit_expression_stmt(self, o: mypy.nodes.ExpressionStmt) -> None:
        self.apply_rules(o)
        super().visit_expression_stmt(o)

    def visit_assignment_stmt(self, o: mypy.nodes.AssignmentStmt) -> None:
        self.apply_rules(o)
        super().visit_assignment_stmt(o)

    def visit_operator_assignment_stmt(self,
                                       o: mypy.nodes.OperatorAssignmentStmt) -> None:
        self.apply_rules(o)
        super().visit_operator_assignment_stmt(o)

    def visit_while_stmt(self, o: mypy.nodes.WhileStmt) -> None:
        self.apply_rules(o)
        super().visit_while_stmt(o)

    def visit_for_stmt(self, o: mypy.nodes.ForStmt) -> None:
        self.apply_rules(o)
        super().visit_for_stmt(o)

    def visit_return_stmt(self, o: mypy.nodes.ReturnStmt) -> None:
        self.apply_rules(o)
        super().visit_return_stmt(o)

    def visit_assert_stmt(self, o: mypy.nodes.AssertStmt) -> None:
        self.apply_rules(o)
        super().visit_assert_stmt(o)

    def visit_del_stmt(self, o: mypy.nodes.DelStmt) -> None:
        self.apply_rules(o)
        super().visit_del_stmt(o)

    def visit_if_stmt(self, o: mypy.nodes.IfStmt) -> None:
        self.apply_rules(o)
        super().visit_if_stmt(o)

    def visit_break_stmt(self, o: mypy.nodes.BreakStmt) -> None:
        self.apply_rules(o)
        super().visit_break_stmt(o)

    def visit_continue_stmt(self, o: mypy.nodes.ContinueStmt) -> None:
        self.apply_rules(o)
        super().visit_continue_stmt(o)

    def visit_pass_stmt(self, o: mypy.nodes.PassStmt) -> None:
        self.apply_rules(o)
        super().visit_pass_stmt(o)

    def visit_raise_stmt(self, o: mypy.nodes.RaiseStmt) -> None:
        self.apply_rules(o)
        super().visit_raise_stmt(o)

    def visit_try_stmt(self, o: mypy.nodes.TryStmt) -> None:
        self.apply_rules(o)
        super().visit_try_stmt(o)

    def visit_with_stmt(self, o: mypy.nodes.WithStmt) -> None:
        self.apply_rules(o)
        super().visit_with_stmt(o)

    def visit_print_stmt(self, o: mypy.nodes.PrintStmt) -> None:
        self.apply_rules(o)
        super().visit_print_stmt(o)

    def visit_exec_stmt(self, o: mypy.nodes.ExecStmt) -> None:
        self.apply_rules(o)
        super().visit_exec_stmt(o)

    # Expressions

    def visit_int_expr(self, o: mypy.nodes.IntExpr) -> None:
        self.apply_rules(o)
        super().visit_int_expr(o)

    def visit_str_expr(self, o: mypy.nodes.StrExpr) -> None:
        self.apply_rules(o)
        super().visit_str_expr(o)

    def visit_bytes_expr(self, o: mypy.nodes.BytesExpr) -> None:
        self.apply_rules(o)
        super().visit_bytes_expr(o)

    def visit_unicode_expr(self, o: mypy.nodes.UnicodeExpr) -> None:
        self.apply_rules(o)
        super().visit_unicode_expr(o)

    def visit_float_expr(self, o: mypy.nodes.FloatExpr) -> None:
        self.apply_rules(o)
        super().visit_float_expr(o)

    def visit_complex_expr(self, o: mypy.nodes.ComplexExpr) -> None:
        self.apply_rules(o)
        super().visit_complex_expr(o)

    def visit_ellipsis(self, o: mypy.nodes.EllipsisExpr) -> None:
        self.apply_rules(o)
        super().visit_ellipsis(o)

    def visit_star_expr(self, o: mypy.nodes.StarExpr) -> None:
        self.apply_rules(o)
        super().visit_star_expr(o)

    def visit_name_expr(self, o: mypy.nodes.NameExpr) -> None:
        self.apply_rules(o)
        super().visit_name_expr(o)

    def visit_member_expr(self, o: mypy.nodes.MemberExpr) -> None:
        self.apply_rules(o)
        super().visit_member_expr(o)

    def visit_yield_from_expr(self, o: mypy.nodes.YieldFromExpr) -> None:
        self.apply_rules(o)
        super().visit_yield_from_expr(o)

    def visit_yield_expr(self, o: mypy.nodes.YieldExpr) -> None:
        self.apply_rules(o)
        super().visit_yield_expr(o)

    def visit_call_expr(self, o: mypy.nodes.CallExpr) -> None:
        self.apply_rules(o)
        super().visit_call_expr(o)

    def visit_op_expr(self, o: mypy.nodes.OpExpr) -> None:
        self.apply_rules(o)
        super().visit_op_expr(o)

    def visit_comparison_expr(self, o: mypy.nodes.ComparisonExpr) -> None:
        self.apply_rules(o)
        super().visit_comparison_expr(o)

    def visit_cast_expr(self, o: mypy.nodes.CastExpr) -> None:
        self.apply_rules(o)
        super().visit_cast_expr(o)

    def visit_reveal_type_expr(self, o: mypy.nodes.RevealTypeExpr) -> None:
        self.apply_rules(o)
        super().visit_reveal_type_expr(o)

    def visit_super_expr(self, o: mypy.nodes.SuperExpr) -> None:
        self.apply_rules(o)
        super().visit_super_expr(o)

    def visit_unary_expr(self, o: mypy.nodes.UnaryExpr) -> None:
        self.apply_rules(o)
        super().visit_unary_expr(o)

    def visit_list_expr(self, o: mypy.nodes.ListExpr) -> None:
        self.apply_rules(o)
        super().visit_list_expr(o)

    def visit_dict_expr(self, o: mypy.nodes.DictExpr) -> None:
        self.apply_rules(o)
        super().visit_dict_expr(o)

    def visit_tuple_expr(self, o: mypy.nodes.TupleExpr) -> None:
        self.apply_rules(o)
        super().visit_tuple_expr(o)

    def visit_set_expr(self, o: mypy.nodes.SetExpr) -> None:
        self.apply_rules(o)
        super().visit_set_expr(o)

    def visit_index_expr(self, o: mypy.nodes.IndexExpr) -> None:
        self.apply_rules(o)
        super().visit_index_expr(o)

    def visit_type_application(self, o: mypy.nodes.TypeApplication) -> None:
        self.apply_rules(o)
        super().visit_type_application(o)

    def visit_func_expr(self, o: mypy.nodes.FuncExpr) -> None:
        self.apply_rules(o)
        super().visit_func_expr(o)

    def visit_list_comprehension(self, o: mypy.nodes.ListComprehension) -> None:
        self.apply_rules(o)
        super().visit_list_comprehension(o)

    def visit_set_comprehension(self, o: mypy.nodes.SetComprehension) -> None:
        self.apply_rules(o)
        super().visit_set_comprehension(o)

    def visit_dictionary_comprehension(self, o: mypy.nodes.DictionaryComprehension) -> None:
        self.apply_rules(o)
        super().visit_dictionary_comprehension(o)

    def visit_generator_expr(self, o: mypy.nodes.GeneratorExpr) -> None:
        self.apply_rules(o)
        super().visit_generator_expr(o)

    def visit_slice_expr(self, o: mypy.nodes.SliceExpr) -> None:
        self.apply_rules(o)
        super().visit_slice_expr(o)

    def visit_conditional_expr(self, o: mypy.nodes.ConditionalExpr) -> None:
        self.apply_rules(o)
        super().visit_conditional_expr(o)

    def visit_backquote_expr(self, o: mypy.nodes.BackquoteExpr) -> None:
        self.apply_rules(o)
        super().visit_backquote_expr(o)

    def visit_type_var_expr(self, o: mypy.nodes.TypeVarExpr) -> None:
        self.apply_rules(o)
        super().visit_type_var_expr(o)

    def visit_type_alias_expr(self, o: mypy.nodes.TypeAliasExpr) -> None:
        self.apply_rules(o)
        super().visit_type_alias_expr(o)

    def visit_namedtuple_expr(self, o: mypy.nodes.NamedTupleExpr) -> None:
        self.apply_rules(o)
        super().visit_namedtuple_expr(o)

    def visit_typeddict_expr(self, o: mypy.nodes.TypedDictExpr) -> None:
        self.apply_rules(o)
        super().visit_typeddict_expr(o)

    def visit_newtype_expr(self, o: mypy.nodes.NewTypeExpr) -> None:
        self.apply_rules(o)
        super().visit_newtype_expr(o)

    def visit__promote_expr(self, o: mypy.nodes.PromoteExpr) -> None:
        self.apply_rules(o)
        super().visit__promote_expr(o)

    def visit_await_expr(self, o: mypy.nodes.AwaitExpr) -> None:
        self.apply_rules(o)
        super().visit_await_expr(o)

    def visit_temp_node(self, o: mypy.nodes.TempNode) -> None:
        self.apply_rules(o)
        super().visit_temp_node(o)


def main(args):
    filename = args[1]
    with open(filename) as input_file:
        ast = parse(input_file.read(), filename, None, Options())

    print(ast)
    ast.accept(RuleVisitor())

if __name__ == '__main__':
    sys.exit(main(sys.argv))
