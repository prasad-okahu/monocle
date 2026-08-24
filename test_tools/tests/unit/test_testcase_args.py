"""Shared testcase-argument resolution for the fluent entry points."""
import pytest

from monocle_test_tools.testcase import Eval, FluentTestCase
from monocle_test_tools.testcase_args import resolve_testcase


def test_dict_is_converted_to_a_model():
    tc = resolve_testcase({"evals": {"hallucination": "minor"}})

    assert isinstance(tc, FluentTestCase)
    assert tc.evals == [Eval(name="hallucination", result="minor")]


def test_model_is_returned_unchanged():
    original = FluentTestCase(input=("go",))

    assert resolve_testcase(original) is original


def test_conflicting_argument_raises_naming_it():
    with pytest.raises(ValueError, match="'eval_name' cannot be combined with 'testcase'"):
        resolve_testcase({"evals": {"a": "x"}}, eval_name="hallucination")


def test_all_conflicting_arguments_are_named():
    with pytest.raises(ValueError, match=r"'eval_name', 'expected'"):
        resolve_testcase({"evals": {"a": "x"}}, eval_name="h", expected="good")


def test_none_valued_arguments_are_not_conflicts():
    tc = resolve_testcase({"evals": {"a": "x"}}, eval_name=None, expected=None)

    assert tc.evals == [Eval(name="a", result="x")]


def test_empty_positional_args_tuple_is_not_a_conflict():
    """run_agent passes its *args through; an empty tuple means none were given."""
    tc = resolve_testcase({"input": "go"}, args=())

    assert tc.input == ("go",)


def test_non_empty_positional_args_tuple_is_a_conflict():
    with pytest.raises(ValueError, match="'args' cannot be combined with 'testcase'"):
        resolve_testcase({"input": "go"}, args=("go",))


def test_wrong_type_raises():
    with pytest.raises(TypeError, match="testcase must be a FluentTestCase or dict"):
        resolve_testcase("not a testcase")
