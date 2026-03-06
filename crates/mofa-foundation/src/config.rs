//! Agent 配置文件解析
//!
//! 支持多种配置格式: YAML, TOML, JSON, INI, RON, JSON5
//!
//! # 示例配置 (agent.yml, agent.toml, agent.json, etc.)
//!
//! ```yaml
//! agent:
//!   id: "my-agent-001"
//!   name: "My LLM Agent"
//!
//! llm:
//!   provider: openai          # openai, ollama, azure
//!   model: gpt-4o
//!   api_key: ${OPENAI_API_KEY}  # 支持环境变量
//!   base_url: null            # 可选，用于自定义 endpoint
//!   temperature: 0.7
//!   max_tokens: 4096
//!   system_prompt: "You are a helpful assistant."
//!
//! tools:
//!   - name: web_search
//!     enabled: true
//!   - name: calculator
//!     enabled: true
//!
//! runtime:
//!   max_concurrent_tasks: 10
//!   default_timeout_secs: 30
//! ```

use mofa_kernel::config::{from_str, load_config};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Agent 配置文件根结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentYamlConfig {
    /// Agent 基本信息
    pub agent: AgentInfo,
    /// LLM 配置
    #[serde(default)]
    pub llm: Option<LLMYamlConfig>,
    /// 工具配置
    #[serde(default)]
    pub tools: Option<Vec<ToolConfig>>,
    /// 运行时配置
    #[serde(default)]
    pub runtime: Option<RuntimeConfig>,
    /// Skills 配置
    #[serde(default)]
    pub skills: Option<SkillsYamlConfig>,
    /// 输入端口
    #[serde(default)]
    pub inputs: Option<Vec<String>>,
    /// 输出端口
    #[serde(default)]
    pub outputs: Option<Vec<String>>,
}

/// Agent 基本信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent ID
    pub id: String,
    /// Agent 名称
    pub name: String,
    /// 描述
    #[serde(default)]
    pub description: Option<String>,
    /// 能力列表
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// LLM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMYamlConfig {
    /// Provider 类型: openai, ollama, azure, compatible, anthropic, gemini
    #[serde(default = "default_provider")]
    pub provider: String,
    /// 模型名称
    #[serde(default)]
    pub model: Option<String>,
    /// API Key (支持 ${ENV_VAR} 语法)
    #[serde(default)]
    pub api_key: Option<String>,
    /// API Base URL
    #[serde(default)]
    pub base_url: Option<String>,
    /// Azure deployment name
    #[serde(default)]
    pub deployment: Option<String>,
    /// 温度参数
    #[serde(default)]
    pub temperature: Option<f32>,
    /// 最大 token 数
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// 系统提示词
    #[serde(default)]
    pub system_prompt: Option<String>,
}

fn default_provider() -> String {
    "openai".to_string()
}

impl Default for LLMYamlConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: None,
            api_key: None,
            base_url: None,
            deployment: None,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            system_prompt: None,
        }
    }
}

/// 工具配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// 工具名称
    pub name: String,
    /// 是否启用
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// 工具特定配置
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

fn default_true() -> bool {
    true
}

/// 运行时配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// 最大并发任务数
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_tasks: usize,
    /// 默认超时（秒）
    #[serde(default = "default_timeout")]
    pub default_timeout_secs: u64,
}

/// Skills 配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillsYamlConfig {
    /// Additional skill search directories.
    #[serde(default)]
    pub search_dirs: Vec<String>,
    /// Skills to always load into context.
    #[serde(default)]
    pub always_load: Vec<String>,
    /// Optional remote hub configuration.
    #[serde(default)]
    pub hub: Option<SkillHubYamlConfig>,
}

/// Remote skills hub configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillHubYamlConfig {
    /// Hub catalog endpoint
    #[serde(default)]
    pub catalog_url: Option<String>,
    /// Auto-install missing explicitly requested skills
    #[serde(default = "default_true")]
    pub auto_install: bool,
    /// Compatibility targets accepted by this agent/runtime
    #[serde(default)]
    pub compatibility_targets: Vec<String>,
}

impl Default for SkillHubYamlConfig {
    fn default() -> Self {
        Self {
            catalog_url: None,
            auto_install: true,
            compatibility_targets: Vec::new(),
        }
    }
}

fn default_max_concurrent() -> usize {
    10
}

fn default_timeout() -> u64 {
    30
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            default_timeout_secs: 30,
        }
    }
}

impl AgentYamlConfig {
    /// 从文件加载配置 (自动检测格式)
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        load_config(&path_str).map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))
    }

    /// 从字符串解析配置 (指定格式)
    pub fn from_str_with_format(content: &str, format: &str) -> anyhow::Result<Self> {
        use config::FileFormat;

        let file_format = match format.to_lowercase().as_str() {
            "yaml" | "yml" => FileFormat::Yaml,
            "toml" => FileFormat::Toml,
            "json" => FileFormat::Json,
            "ini" => FileFormat::Ini,
            "ron" => FileFormat::Ron,
            "json5" => FileFormat::Json5,
            _ => return Err(anyhow::anyhow!("Unsupported config format: {}", format)),
        };

        from_str(content, file_format).map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
    }

    /// 从字符串解析配置 (自动检测为 YAML)
    pub fn parse(content: &str) -> anyhow::Result<Self> {
        Self::from_str_with_format(content, "yaml")
    }
}

/// Return the global MoFA config directory.
pub fn mofa_config_dir() -> anyhow::Result<std::path::PathBuf> {
    let config_dir = dirs_next::config_dir()
        .ok_or_else(|| anyhow::anyhow!("Unable to determine config directory"))?;
    Ok(config_dir.join("mofa"))
}

/// Load global MoFA configuration as a flat string map.
pub fn load_global_config() -> anyhow::Result<HashMap<String, String>> {
    load_global_config_from_dir(mofa_config_dir()?)
}

/// Load global MoFA configuration from an explicit config directory.
pub fn load_global_config_from_dir(
    config_dir: impl AsRef<Path>,
) -> anyhow::Result<HashMap<String, String>> {
    let config_file = config_dir.as_ref().join("config.yml");
    if !config_file.exists() {
        return Ok(HashMap::new());
    }

    let config: serde_json::Value = load_config(config_file.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    let mut result = HashMap::new();
    flatten_config_value(None, &config, &mut result);

    Ok(result)
}

pub fn parse_global_string_list(value: &str) -> Vec<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if trimmed.starts_with('[')
        && let Ok(values) = serde_json::from_str::<Vec<String>>(trimmed)
    {
        return values
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();
    }

    trimmed
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn flatten_config_value(
    prefix: Option<&str>,
    value: &serde_json::Value,
    output: &mut HashMap<String, String>,
) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, value) in object {
                let next_key = match prefix {
                    Some(prefix) => format!("{prefix}.{key}"),
                    None => key.clone(),
                };
                flatten_config_value(Some(&next_key), value, output);
            }
        }
        _ => {
            if let Some(prefix) = prefix {
                let rendered = value
                    .as_str()
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| value.to_string());
                output.insert(prefix.to_string(), rendered);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{LazyLock, Mutex};

    static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn test_parse_skills_config_from_yaml() {
        let config = AgentYamlConfig::parse(
            r#"
agent:
  id: skill-agent
  name: Skill Agent
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
"#,
        )
        .unwrap();

        let skills = config.skills.unwrap();
        assert_eq!(skills.search_dirs, vec!["./skills"]);
        assert_eq!(skills.always_load, vec!["catalog"]);

        let hub = skills.hub.unwrap();
        assert_eq!(
            hub.catalog_url.as_deref(),
            Some("https://clawhub.run/api/skills/catalog")
        );
        assert!(hub.auto_install);
        assert_eq!(hub.compatibility_targets, vec!["mofa", "universal"]);
    }

    #[test]
    fn test_load_global_config_reads_dotted_skill_keys() {
        let _env_guard = ENV_MUTEX
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let temp = tempfile::TempDir::new().unwrap();
        let config_home = temp.path().join("xdg-config");
        std::fs::create_dir_all(config_home.join("mofa")).unwrap();
        std::fs::write(
            config_home.join("mofa").join("config.yml"),
            "skills.hub.catalog_url: https://hub.example/catalog\nskills.hub.auto_install: true\nskills.hub.compatibility_targets: mofa,portable\n",
        )
        .unwrap();

        let previous_config = std::env::var_os("XDG_CONFIG_HOME");
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", &config_home);
        }

        let loaded = load_global_config().unwrap();
        assert_eq!(
            loaded.get("skills.hub.catalog_url").map(String::as_str),
            Some("https://hub.example/catalog")
        );
        assert_eq!(
            loaded.get("skills.hub.auto_install").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            loaded
                .get("skills.hub.compatibility_targets")
                .map(String::as_str),
            Some("mofa,portable")
        );

        match previous_config {
            Some(value) => unsafe { std::env::set_var("XDG_CONFIG_HOME", value) },
            None => unsafe { std::env::remove_var("XDG_CONFIG_HOME") },
        }
    }

    #[test]
    fn test_parse_global_string_list_supports_json_array_strings() {
        assert_eq!(
            parse_global_string_list(r#"["mofa","portable"]"#),
            vec!["mofa", "portable"]
        );
    }

    #[test]
    fn test_load_global_config_flattens_nested_skill_lists_as_json_arrays() {
        let temp = tempfile::TempDir::new().unwrap();
        let config_dir = temp.path().join("mofa");
        std::fs::create_dir_all(&config_dir).unwrap();
        std::fs::write(
            config_dir.join("config.yml"),
            r#"
skills:
  hub:
    catalog_url: https://hub.example/catalog
    auto_install: true
    compatibility_targets:
      - mofa
      - portable
"#,
        )
        .unwrap();

        let loaded = load_global_config_from_dir(&config_dir).unwrap();
        let parsed = parse_global_string_list(
            loaded
                .get("skills.hub.compatibility_targets")
                .expect("compatibility target list should be present"),
        );
        assert_eq!(parsed, vec!["mofa", "portable"]);
    }
}
