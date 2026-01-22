# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/opper-ai/opperai-agent-sdk/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/opper-ai/opperai-agent-sdk/releases/tag/v0.1.0
