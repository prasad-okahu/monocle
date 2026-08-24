"""Argument resolution shared by the fluent entry points that accept ``testcase=``.

These are pure functions over a ``FluentTestCase`` with no ``TraceAssertion``
dependency, so they stay testable on their own and are reusable by the span
selectors when those gain ``testcase=`` support.
"""
from typing import Any, Union

from monocle_test_tools.testcase import FluentTestCase


def resolve_testcase(testcase: Union[FluentTestCase, dict], **forbidden: Any) -> FluentTestCase:
    """Normalize *testcase* to a model, rejecting arguments it conflicts with.

    A test case already carries the values the *forbidden* arguments would
    supply, so a caller passing both has written a test whose intent is
    ambiguous -- that is a mistake in the test, not a case to silently resolve
    by precedence.

    Args:
        testcase: A FluentTestCase, or a dict in any shape it accepts.
        **forbidden: Arguments that must not accompany a test case, by name. An
            argument counts as given when it is neither None nor an empty tuple,
            so a caller can forward its ``*args`` directly.

    Returns:
        The test case as a FluentTestCase.

    Raises:
        ValueError: If any forbidden argument was given.
        TypeError: If *testcase* is neither a FluentTestCase nor a dict.
    """
    given = [name for name, value in forbidden.items()
             if value is not None and value != ()]
    if given:
        names = ", ".join(f"'{name}'" for name in sorted(given))
        raise ValueError(
            f"{names} cannot be combined with 'testcase'; the test case already "
            "supplies these values")

    if isinstance(testcase, FluentTestCase):
        return testcase
    if isinstance(testcase, dict):
        return FluentTestCase.model_validate(testcase)
    raise TypeError(
        f"testcase must be a FluentTestCase or dict, got {type(testcase).__name__}")
