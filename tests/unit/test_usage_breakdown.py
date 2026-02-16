"""Tests for Cost, BaseUsage, Usage breakdown, and usage tracking methods."""

from opper_agents.base.context import (
    Cost,
    BaseUsage,
    Usage,
    AgentContext,
)


def test_cost_defaults():
    cost = Cost()
    assert cost.generation == 0.0
    assert cost.platform == 0.0
    assert cost.total == 0.0


def test_cost_add():
    a = Cost(generation=1.0, platform=0.5, total=1.5)
    b = Cost(generation=2.0, platform=1.0, total=3.0)
    c = a.add(b)
    assert c.generation == 3.0
    assert c.platform == 1.5
    assert c.total == 4.5


def test_base_usage_defaults():
    usage = BaseUsage()
    assert usage.requests == 0
    assert usage.total_tokens == 0
    assert usage.cost.total == 0.0


def test_base_usage_add():
    a = BaseUsage(requests=1, total_tokens=100, cost=Cost(total=0.01))
    b = BaseUsage(requests=2, total_tokens=200, cost=Cost(total=0.02))
    c = a.add(b)
    assert c.requests == 3
    assert c.total_tokens == 300
    assert c.cost.total == 0.03


def test_usage_defaults():
    usage = Usage()
    assert usage.requests == 0
    assert usage.breakdown is None
    assert usage.cost.total == 0.0


def test_usage_add_without_breakdown():
    a = Usage(requests=1, total_tokens=100)
    b = Usage(requests=2, total_tokens=200)
    c = a.add(b)
    assert c.requests == 3
    assert c.total_tokens == 300
    assert c.breakdown is None


def test_usage_add_with_breakdown():
    a = Usage(
        requests=1,
        total_tokens=100,
        breakdown={"agent_a": BaseUsage(requests=1, total_tokens=100)},
    )
    b = Usage(
        requests=2,
        total_tokens=200,
        breakdown={"agent_b": BaseUsage(requests=2, total_tokens=200)},
    )
    c = a.add(b)
    assert c.requests == 3
    assert c.breakdown is not None
    assert "agent_a" in c.breakdown
    assert "agent_b" in c.breakdown
    assert c.breakdown["agent_a"].total_tokens == 100
    assert c.breakdown["agent_b"].total_tokens == 200


def test_usage_add_merges_same_source_breakdown():
    a = Usage(
        requests=1,
        total_tokens=100,
        breakdown={"agent_a": BaseUsage(requests=1, total_tokens=100)},
    )
    b = Usage(
        requests=1,
        total_tokens=50,
        breakdown={"agent_a": BaseUsage(requests=1, total_tokens=50)},
    )
    c = a.add(b)
    assert c.breakdown is not None
    assert c.breakdown["agent_a"].requests == 2
    assert c.breakdown["agent_a"].total_tokens == 150


def test_usage_to_base_usage():
    usage = Usage(
        requests=5,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost=Cost(generation=0.1, platform=0.01, total=0.11),
    )
    base = usage.to_base_usage()
    assert isinstance(base, BaseUsage)
    assert base.requests == 5
    assert base.total_tokens == 150
    assert base.cost.total == 0.11


def test_context_update_usage_with_source():
    ctx = AgentContext(agent_name="Test")
    delta = Usage(requests=1, total_tokens=100)
    ctx.update_usage_with_source("child_agent", delta)

    assert ctx.usage.requests == 1
    assert ctx.usage.total_tokens == 100
    assert ctx.usage.breakdown is not None
    assert "child_agent" in ctx.usage.breakdown
    assert ctx.usage.breakdown["child_agent"].total_tokens == 100


def test_context_update_usage_with_source_accumulates():
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage_with_source("child", Usage(requests=1, total_tokens=100))
    ctx.update_usage_with_source("child", Usage(requests=1, total_tokens=50))

    assert ctx.usage.requests == 2
    assert ctx.usage.total_tokens == 150
    assert ctx.usage.breakdown is not None
    assert ctx.usage.breakdown["child"].requests == 2
    assert ctx.usage.breakdown["child"].total_tokens == 150


def test_context_update_usage_with_source_multiple_sources():
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage_with_source("agent_a", Usage(requests=1, total_tokens=100))
    ctx.update_usage_with_source("agent_b", Usage(requests=2, total_tokens=200))

    assert ctx.usage.breakdown is not None
    assert len(ctx.usage.breakdown) == 2
    assert ctx.usage.breakdown["agent_a"].total_tokens == 100
    assert ctx.usage.breakdown["agent_b"].total_tokens == 200


def test_cleanup_breakdown_if_only_parent():
    ctx = AgentContext(agent_name="Parent")
    ctx.update_usage_with_source("Parent", Usage(requests=1, total_tokens=100))

    assert ctx.usage.breakdown is not None
    ctx.cleanup_breakdown_if_only_parent("Parent")
    assert ctx.usage.breakdown is None


def test_cleanup_breakdown_preserves_multiple_sources():
    ctx = AgentContext(agent_name="Parent")
    ctx.update_usage_with_source("Parent", Usage(requests=1, total_tokens=100))
    ctx.update_usage_with_source("Child", Usage(requests=1, total_tokens=50))

    ctx.cleanup_breakdown_if_only_parent("Parent")
    # Should NOT be cleaned up because there are 2 sources
    assert ctx.usage.breakdown is not None
    assert len(ctx.usage.breakdown) == 2


def test_cleanup_breakdown_empty():
    ctx = AgentContext(agent_name="Parent")
    # No breakdown at all
    ctx.cleanup_breakdown_if_only_parent("Parent")
    assert ctx.usage.breakdown is None


def test_usage_add_merges_cost():
    a = Usage(
        requests=1,
        total_tokens=100,
        cost=Cost(generation=0.01, platform=0.005, total=0.015),
    )
    b = Usage(
        requests=1,
        total_tokens=200,
        cost=Cost(generation=0.02, platform=0.01, total=0.03),
    )
    c = a.add(b)
    assert c.cost.generation == 0.03
    assert c.cost.platform == 0.015
    assert c.cost.total == 0.045


def test_cost_propagated_through_breakdown():
    """Cost should accumulate through update_usage_with_source."""
    ctx = AgentContext(agent_name="Parent")
    delta = Usage(
        requests=1,
        total_tokens=100,
        cost=Cost(generation=0.01, platform=0.005, total=0.015),
    )
    ctx.update_usage_with_source("child", delta)
    assert ctx.usage.cost.generation == 0.01
    assert ctx.usage.cost.total == 0.015
    assert ctx.usage.breakdown is not None
    assert ctx.usage.breakdown["child"].cost.generation == 0.01
