from app.llm_provider import LocalRuleBasedProvider


def _provider():
    return LocalRuleBasedProvider()


def test_math_route_requires_digit_on_both_sides_of_operator():
    p = _provider()
    assert p.classify_route("What is 12 * (3 + 4) / 2?", []) == "math"
    assert p.classify_route("calculate 9 - 4", []) == "math"


def test_hyphenated_compound_does_not_false_positive_to_math():
    # Regression test: "90-minute" contains a digit immediately followed by a
    # hyphen, which an earlier, looser regex misrouted into the math agent.
    p = _provider()
    assert p.classify_route("I like 90-minute focus blocks.", []) == "general"
    assert p.classify_route("Give me a 3-step plan.", []) != "math"


def test_research_route_on_question_words():
    p = _provider()
    assert p.classify_route("What is the capital of France?", []) == "research"
    assert p.classify_route("Explain how photosynthesis works.", []) == "research"


def test_escalation_route_on_sensitive_keywords():
    p = _provider()
    assert p.classify_route("Please delete my account.", []) == "escalate"
    assert p.classify_route("I need a refund processed today.", []) == "escalate"


def test_escalation_takes_priority_over_math_pattern():
    p = _provider()
    # Contains both a "digit op digit"-style substring AND an escalation
    # keyword - escalation must win, since safety review should never be
    # skipped because a request also happens to contain numbers.
    assert p.classify_route("Refund $10 - $2 in fees please.", []) == "escalate"


def test_general_route_as_fallback():
    p = _provider()
    assert p.classify_route("I enjoy going for long walks.", []) == "general"


def test_respond_echoes_tool_output_verbatim():
    p = _provider()
    assert p.respond("2+2?", tool_output="2 + 2 = 4") == "2 + 2 = 4"


def test_respond_falls_back_to_context_when_no_tool_output():
    p = _provider()
    result = p.respond("anything", context=["Q: old question\nA: old answer here"])
    assert "old answer here" in result
    assert not result.startswith("Q:")  # the Q: half should be stripped, not echoed


def test_respond_default_message_when_nothing_available():
    p = _provider()
    result = p.respond("hello there")
    assert "hello there" in result
