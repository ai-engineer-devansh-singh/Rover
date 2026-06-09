# Rover

Local analytics for AI coding sessions. Understand how you build with AI.

## Install

**From source (not yet on PyPI):**

```bash
git clone https://github.com/ai-engineer-devansh-singh/Rover.git
cd Rover
pip install -e .
```

Requires Python 3.10+.

## Quick Start

### One-liners

**Ollama (free, local):**
```bash
ollama pull llama3.2 && rover analyze --since 2m
```

**OpenAI:**
```bash
export ROVER_LLM_API_KEY="sk-your-key"; export ROVER_LLM_BASE_URL="https://api.openai.com/v1"; rover analyze --since 2m
```

**Anthropic:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"; rover analyze --since 2m
```

**OpenRouter:**
```bash
export ROVER_LLM_API_KEY="your-key"; export ROVER_LLM_BASE_URL="https://openrouter.ai/api/v1"; rover analyze --since 2m
```

**No LLM (free, offline):**
```bash
rover analyze --since 2m --no-llm
```

Report auto-opens in your browser.

## Supported Agents

| Agent | Transcript Location |
|-------|---------------------|
| Claude Code | `~/.claude/projects/` |
| Cursor | `~/.config/Cursor/User/workspaceStorage/` |
| Codex CLI | `~/.codex/sessions/` |

## CLI Options

```bash
rover analyze              # Scan all sessions (default)
rover analyze --project X  # Specific repo only
rover analyze --since 7d   # Time window: 6h, 7d, 2w, 1m
rover analyze --no-llm     # Rule-based (free, offline)
rover analyze --no-open    # Don't auto-open browser
rover config               # Check LLM provider status
```

## Custom Paths

```bash
rover analyze --claude-path /mnt/backup/claude-projects
rover analyze --cursor-path /data/cursor-workspaces
rover analyze --codex-path /shared/codex-sessions
```

## Privacy

- Your API key, your LLM. No proxy, no upload.
- File bodies stay local. Only redacted summaries go to your LLM.
- SQLite cache at `~/.rover/cache/`.
- No telemetry. No phone home.

## Report Includes

- 5-dimension builder scores
- Archetype classification
- Session timeline
- Tool usage patterns
- Work streams
- Growth recommendations with examples

## License

MIT
