# Skills System

MoFA's skills system enables progressive disclosure of capabilities to manage context length and cost.

## Overview

The skills system:
- **Reduces context** by loading only skill summaries initially
- **On-demand loading** of full skill content when needed
- **Multi-directory search** with priority ordering
- **Remote hub installation** for pulling compatible skills into the local MoFA data directory
- **Managed hub registry** for install/update/remove/list operations
- **Runtime context integration** for explicit requested-skill loading through `LLMAgent`

## Using Skills

```rust
use mofa_sdk::skills::SkillsManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize skills manager
    let skills = SkillsManager::new("./skills")?;

    // Build summary for context injection
    let summary = skills.build_skills_summary().await;

    // Load specific skills on demand
    let requested = vec!["pdf_processing".to_string()];
    let content = skills.load_skills_for_context(&requested).await;

    // Inject into prompt
    let system_prompt = format!(
        "You are a helpful assistant.\n\n# Skills Summary\n{}\n\n# Requested Skills\n{}",
        summary, content
    );

    Ok(())
}
```

## Skill Definition

Create a `SKILL.md` file in your skills directory:

```markdown
# PDF Processing

## Summary
Extract text, tables, and images from PDF documents.

## Capabilities
- Text extraction with layout preservation
- Table detection and extraction
- Image extraction
- Metadata reading

## Usage
```
extract_pdf(path: str) -> PDFContent
```

## Examples
- Extract invoice data: `extract_pdf("invoice.pdf")`
```

## Skill Directory Structure

```
skills/
├── pdf_processing/
│   └── SKILL.md
├── web_search/
│   └── SKILL.md
└── data_analysis/
    └── SKILL.md
```

## Remote Hub Workflow

MoFA can search and install skills from a hub-style remote catalog.

```bash
# Search the default hub catalog
mofa skill search pdf

# List available categories
mofa skill categories

# Show a skill's metadata
mofa skill info pdf_processing

# Sync the remote catalog into the local cache
mofa skill sync

# Install the latest version of a skill
mofa skill install pdf_processing

# Install an explicit version
mofa skill install pdf_processing@1.2.0
# or
mofa skill install pdf_processing --skill-version 1.2.0

# Show locally installed hub skills
mofa skill list

# Update one installed skill
mofa skill update pdf_processing

# Update all installed hub skills
mofa skill update --all

# Remove an installed hub skill
mofa skill remove pdf_processing

# Inspect or clear the local hub cache
mofa skill cache status
mofa skill cache clear
```

Hub configuration resolves differently depending on the entrypoint:
- CLI skill commands: explicit `--catalog-url` (or `MOFA_SKILLS_HUB_CATALOG_URL` via clap), then `mofa config`, then defaults
- Agent/runtime loading from `LLMAgentBuilder::from_config_file(...)`: agent YAML `skills.hub.*`, then `MOFA_SKILLS_HUB_CATALOG_URL` / `MOFA_SKILLS_HUB_AUTO_INSTALL`, then `mofa config`, then defaults
- In-memory `LLMAgentBuilder::from_yaml_config(...)`: agent YAML `skills.hub.*`, then defaults

Examples:

```bash
mofa config set skills.hub.catalog_url https://clawhub.run/api/skills/catalog
mofa config set skills.hub.auto_install true
mofa config set skills.hub.compatibility_targets mofa,universal,portable
```

Authentication tokens are read from environment variables only:
- `MOFA_SKILLS_HUB_TOKEN`
- `CLAWHUB_AUTH_TOKEN`
- `CLAWHUB_API_KEY`

Installed hub skills are written under the standard MoFA data directory:

- Linux: `~/.local/share/mofa/skills/hub/<skill-name>/`
- macOS: `~/Library/Application Support/mofa/skills/hub/<skill-name>/`
- Windows: `%LOCALAPPDATA%\\mofa\\skills\\hub\\<skill-name>\\`

Catalog cache data is stored separately so search and info commands can fall back to offline results after a successful `mofa skill sync`.

## Agent YAML Integration

You can wire the skills hub into an agent config directly:

```yaml
agent:
  id: skill-agent
  name: Skill Agent

llm:
  provider: compatible
  model: local-model
  base_url: http://localhost:11434/v1

skills:
  search_dirs:
    - ./skills
  always_load:
    - catalog
  hub:
    catalog_url: https://clawhub.run/api/skills/catalog
    auto_install: true
    compatibility_targets:
      - mofa
      - universal
```

From SDK code, use the skill-aware config loader and explicit requested-skill APIs:

```rust
use mofa_sdk::llm::{builder_from_config_with_skills, LLMAgent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent: LLMAgent = builder_from_config_with_skills("agent.yml")?
        .try_build()?;

    let requested = vec!["pdf_processing".to_string()];
    let answer = agent.ask_with_skills("Summarize this PDF workflow", &requested).await?;
    println!("{}", answer);

    Ok(())
}
```

When `auto_install` is enabled, explicitly requested skills are installed from the hub on first use and then behave like local skills.

## Search Priority

Skills are searched in this order:
1. Workspace skills (project-specific)
2. Built-in skills (framework-provided)
3. System skills (global)

## See Also

- [Tool Development](tool-development.md) — Creating tools
- [Agents](../concepts/agents.md) — Agent concepts
