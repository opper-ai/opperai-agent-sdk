# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-02-16

### Added
- `run()` method on all agents, returning `RunResult(result, usage)` with full usage statistics
- `Cost` model with `generation`, `platform`, and `total` fields — populated from Opper API responses
- `BaseUsage` model for per-source usage breakdown in nested agent scenarios
- Usage `breakdown` field tracking per-source token usage when agents call sub-agents
- `update_usage_with_source()` and `cleanup_breakdown_if_only_parent()` on `AgentContext`
- `on()`, `once()`, `off()` hook methods on agents and `HookManager` for EventEmitter-style hook management
- `MEMORY_READ`, `MEMORY_WRITE`, `MEMORY_ERROR` hook events for memory operation observability
- `tool_call_id` (UUID) in `tool_call` and `tool_result` hook payloads for async correlation
- `output_schema` and `examples` fields on tools and `@tool` decorator for richer tool definitions
- `parallel_tool_execution` parameter on agents to run multiple tool calls concurrently via `asyncio.gather`
- `model` parameter now accepts a list of strings for fallback routing (e.g. `["openai/gpt-4o", "anthropic/claude-3.7-sonnet"]`)
- Deferred span updates: span writes are batched and flushed at end of run for better performance
- Usage propagation from nested agents via `as_tool()` — parent agents track sub-agent token usage
- Exported `RunResult`, `Usage`, `BaseUsage`, `Cost` from `opper_agents`

### Changed
- `_track_usage()` now uses `update_usage_with_source()` for per-agent breakdown tracking
- Span updates in `_execute_tool()` and `_think()` are now deferred and flushed in batch
- Memory operations wrapped in try/except with hook-based error reporting

### Deprecated
- `ReactAgent` — use `Agent` instead (emits `DeprecationWarning`)
- `ChatAgent` — use `Agent` instead (emits `DeprecationWarning`)

### Breaking Changes
- Hook functions registered for `tool_call` and `tool_result` events now receive a `tool_call_id` keyword argument. Hook functions that do not accept `**kwargs` will raise `TypeError`. **Migration:** Add `**kwargs` to your hook function signatures.
- The `model` parameter type widened from `Optional[str]` to `Optional[Union[str, List[str]]]` for fallback routing. Code that performs string operations on `agent.model` (e.g. `.split()`, `isinstance(agent.model, str)`) may need updating. **Migration:** Check `isinstance(agent.model, str)` before string operations.

### Fixed
- Example hook functions now accept `**kwargs` for forward compatibility with new hook parameters

## [0.3.0] - 2026-01-22

### Added
- `agent_tool_timeout` parameter for configurable timeouts when agents run as tools (default: 120s)
- Span types with emojis for better observability (agent, tool, memory)
- Span timing and hierarchy tracking for tools and memory operations
- User message display in streaming example

### Changed
- Removed separate `final_response` step - result now returned directly from the last think step
- Improved think span names with tool and memory types
- Better span hierarchy with proper parent-child relationships

### Fixed
- Move duration to meta in span updates

## [0.2.0] - 2025-10-21

### Added
- Streaming support for agent responses
- Mermaid diagram support for agent visualization
- User agent header for API requests

## [0.1.0] - 2025-10-13

### Added
- Initial release of Opper Agent Python SDK
- Core `Agent` class for building AI agents
- `ReactAgent` for ReAct-style reasoning
- `ChatAgent` for conversational agents
- Tool system with `@tool` decorator
- Hook system for lifecycle events
- Memory management for agent context
- MCP (Model Context Protocol) integration
- Full type annotations with Pydantic models

[Unreleased]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/opper-ai/opperai-agent-sdk/releases/tag/v0.1.0
