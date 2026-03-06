use async_trait::async_trait;
use axum::{Json, Router, routing::get};
use mofa_sdk::config::AgentYamlConfig;
use mofa_sdk::llm::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, FinishReason, LLMProvider,
    LLMResult, MessageContent, Usage,
};
use serde_json::json;
use std::ffi::OsString;
use std::sync::{Arc, LazyLock, Mutex};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::Mutex as AsyncMutex;

static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn restore_env_var(name: &str, value: Option<OsString>) {
    match value {
        Some(value) => unsafe { std::env::set_var(name, value) },
        None => unsafe { std::env::remove_var(name) },
    }
}

struct RecordingProvider {
    last_request: Arc<AsyncMutex<Option<ChatCompletionRequest>>>,
}

impl RecordingProvider {
    fn new() -> Self {
        Self {
            last_request: Arc::new(AsyncMutex::new(None)),
        }
    }

    async fn take_request(&self) -> ChatCompletionRequest {
        self.last_request
            .lock()
            .await
            .clone()
            .expect("request should be recorded")
    }
}

#[async_trait]
impl LLMProvider for RecordingProvider {
    fn name(&self) -> &str {
        "recording"
    }

    fn default_model(&self) -> &str {
        "recording-model"
    }

    async fn chat(&self, request: ChatCompletionRequest) -> LLMResult<ChatCompletionResponse> {
        *self.last_request.lock().await = Some(request);

        Ok(ChatCompletionResponse {
            id: "resp-1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "recording-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant("ok"),
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
            system_fingerprint: None,
        })
    }
}

fn system_text(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .first()
        .and_then(|message| match &message.content {
            Some(MessageContent::Text(text)) => Some(text.clone()),
            _ => None,
        })
        .expect("system prompt should be present")
}

#[tokio::test]
async fn test_builder_from_config_with_skills_auto_installs_requested_skill() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "tags": ["pdf"],
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["mofa"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
    }

    let config_path = temp.path().join("agent.yml");
    std::fs::write(
        &config_path,
        format!(
            r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    catalog_url: http://{address}/catalog
    auto_install: true
"#
        ),
    )
    .unwrap();

    let provider = Arc::new(RecordingProvider::new());
    let requested = vec!["pdf_processing".to_string()];
    let agent = mofa_sdk::llm::builder_from_config_with_skills(&config_path)
        .unwrap()
        .with_provider(provider.clone())
        .try_build()
        .unwrap();

    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("# PDF"));
    assert!(system_prompt.contains("pdf_processing"));
    assert!(
        data_home
            .join("mofa")
            .join("skills")
            .join("hub")
            .join("pdf_processing")
            .join("SKILL.md")
            .exists()
    );

    match previous_data {
        Some(value) => unsafe { std::env::set_var("XDG_DATA_HOME", value) },
        None => unsafe { std::env::remove_var("XDG_DATA_HOME") },
    }
    match previous_cache {
        Some(value) => unsafe { std::env::set_var("XDG_CACHE_HOME", value) },
        None => unsafe { std::env::remove_var("XDG_CACHE_HOME") },
    }

    server.abort();
}

#[tokio::test]
async fn test_standard_from_config_file_honors_skills_section() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "tags": ["pdf"],
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["mofa"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
    }

    let config_path = temp.path().join("agent.yml");
    std::fs::write(
        &config_path,
        format!(
            r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    catalog_url: http://{address}/catalog
    auto_install: true
"#
        ),
    )
    .unwrap();

    let provider = Arc::new(RecordingProvider::new());
    let requested = vec!["pdf_processing".to_string()];
    let agent = mofa_sdk::llm::LLMAgentBuilder::from_config_file(&config_path)
        .unwrap()
        .with_provider(provider.clone())
        .try_build()
        .unwrap();

    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("# PDF"));
    assert!(system_prompt.contains("pdf_processing"));

    restore_env_var("XDG_DATA_HOME", previous_data);
    restore_env_var("XDG_CACHE_HOME", previous_cache);

    server.abort();
}

#[tokio::test]
async fn test_from_yaml_config_honors_skills_section() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["mofa"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
    }

    let config_yaml = format!(
        r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    catalog_url: http://{address}/catalog
    auto_install: true
"#
    );
    let config_path = temp.path().join("from-yaml-agent.yml");
    std::fs::write(&config_path, config_yaml).unwrap();
    let config = AgentYamlConfig::from_file(&config_path).unwrap();

    let provider = Arc::new(RecordingProvider::new());
    let requested = vec!["pdf_processing".to_string()];
    let agent = mofa_sdk::llm::LLMAgentBuilder::from_yaml_config(config)
        .unwrap()
        .with_provider(provider.clone())
        .try_build()
        .unwrap();

    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("# PDF"));
    assert!(system_prompt.contains("pdf_processing"));

    restore_env_var("XDG_DATA_HOME", previous_data);
    restore_env_var("XDG_CACHE_HOME", previous_cache);

    server.abort();
}

#[tokio::test]
async fn test_builder_paths_use_persisted_hub_config_equivalently() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["mofa"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    let config_home = temp.path().join("xdg-config");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();
    std::fs::create_dir_all(config_home.join("mofa")).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    let previous_config = std::env::var_os("XDG_CONFIG_HOME");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
    }

    std::fs::write(
        config_home.join("mofa").join("config.yml"),
        format!(
            "skills.hub.catalog_url: http://{address}/catalog\nskills.hub.auto_install: true\n"
        ),
    )
    .unwrap();

    let config_path = temp.path().join("agent.yml");
    std::fs::write(
        &config_path,
        r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    auto_install: true
"#,
    )
    .unwrap();

    let requested = vec!["pdf_processing".to_string()];

    let provider_a = Arc::new(RecordingProvider::new());
    let agent_a = mofa_sdk::llm::builder_from_config_with_skills(&config_path)
        .unwrap()
        .with_provider(provider_a.clone())
        .try_build()
        .unwrap();

    let provider_b = Arc::new(RecordingProvider::new());
    let agent_b = mofa_sdk::llm::LLMAgentBuilder::from_config_file(&config_path)
        .unwrap()
        .with_provider(provider_b.clone())
        .try_build()
        .unwrap();

    let response_a = agent_a.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response_a, "ok");
    let prompt_a = system_text(&provider_a.take_request().await);
    assert!(prompt_a.contains("# PDF"));

    let response_b = agent_b.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response_b, "ok");
    let prompt_b = system_text(&provider_b.take_request().await);
    assert!(prompt_b.contains("# PDF"));
    assert!(prompt_a.contains("pdf_processing"));
    assert!(prompt_b.contains("pdf_processing"));

    restore_env_var("XDG_DATA_HOME", previous_data);
    restore_env_var("XDG_CACHE_HOME", previous_cache);
    restore_env_var("XDG_CONFIG_HOME", previous_config);

    server.abort();
}

#[tokio::test]
async fn test_from_config_file_accepts_nested_yaml_global_hub_config() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "portable_skill",
                            "description": "Portable skill",
                            "author": "OpenClaw",
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["portable-skill"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: portable_skill\ndescription: Portable skill\n---\n# Portable\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    let config_home = temp.path().join("xdg-config");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();
    std::fs::create_dir_all(config_home.join("mofa")).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    let previous_config = std::env::var_os("XDG_CONFIG_HOME");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
    }

    std::fs::write(
        config_home.join("mofa").join("config.yml"),
        format!(
            r#"
skills:
  hub:
    catalog_url: http://{address}/catalog
    auto_install: true
    compatibility_targets:
      - mofa
      - portable
"#
        ),
    )
    .unwrap();

    let config_path = temp.path().join("agent.yml");
    std::fs::write(
        &config_path,
        r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    auto_install: true
"#,
    )
    .unwrap();

    let provider = Arc::new(RecordingProvider::new());
    let requested = vec!["portable_skill".to_string()];
    let agent = mofa_sdk::llm::LLMAgentBuilder::from_config_file(&config_path)
        .unwrap()
        .with_provider(provider.clone())
        .try_build()
        .unwrap();

    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let system_prompt = system_text(&provider.take_request().await);
    assert!(system_prompt.contains("# Portable"));

    restore_env_var("XDG_DATA_HOME", previous_data);
    restore_env_var("XDG_CACHE_HOME", previous_cache);
    restore_env_var("XDG_CONFIG_HOME", previous_config);

    server.abort();
}

#[tokio::test]
async fn test_from_config_file_honors_env_hub_catalog_override() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["mofa"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    let config_home = temp.path().join("xdg-config");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();
    std::fs::create_dir_all(&config_home).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    let previous_config = std::env::var_os("XDG_CONFIG_HOME");
    let previous_catalog_url = std::env::var_os("MOFA_SKILLS_HUB_CATALOG_URL");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
        std::env::set_var(
            "MOFA_SKILLS_HUB_CATALOG_URL",
            format!("http://{address}/catalog"),
        );
    }

    let config_path = temp.path().join("agent.yml");
    std::fs::write(
        &config_path,
        r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    auto_install: true
"#,
    )
    .unwrap();

    let provider = Arc::new(RecordingProvider::new());
    let requested = vec!["pdf_processing".to_string()];
    let agent = mofa_sdk::llm::LLMAgentBuilder::from_config_file(&config_path)
        .unwrap()
        .with_provider(provider.clone())
        .try_build()
        .unwrap();

    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let system_prompt = system_text(&provider.take_request().await);
    assert!(system_prompt.contains("# PDF"));

    restore_env_var("XDG_DATA_HOME", previous_data);
    restore_env_var("XDG_CACHE_HOME", previous_cache);
    restore_env_var("XDG_CONFIG_HOME", previous_config);
    restore_env_var("MOFA_SKILLS_HUB_CATALOG_URL", previous_catalog_url);

    server.abort();
}

#[tokio::test]
async fn test_from_yaml_config_ignores_persisted_global_hub_settings() {
    let _env_guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["mofa-compatible"]
                        }
                    ]
                }))
            }),
        )
        .route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    let data_home = temp.path().join("xdg-data");
    let cache_home = temp.path().join("xdg-cache");
    let config_home = temp.path().join("xdg-config");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&cache_home).unwrap();
    std::fs::create_dir_all(config_home.join("mofa")).unwrap();

    let previous_data = std::env::var_os("XDG_DATA_HOME");
    let previous_cache = std::env::var_os("XDG_CACHE_HOME");
    let previous_config = std::env::var_os("XDG_CONFIG_HOME");
    unsafe {
        std::env::set_var("XDG_DATA_HOME", &data_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
    }

    std::fs::write(
        config_home.join("mofa").join("config.yml"),
        "skills.hub.compatibility_targets: no-match\n",
    )
    .unwrap();

    let config_yaml = format!(
        r#"
agent:
  id: skill-agent
  name: Skill Agent
skills:
  hub:
    catalog_url: http://{address}/catalog
    auto_install: true
"#
    );
    let config_path = temp.path().join("from-yaml-agent.yml");
    std::fs::write(&config_path, config_yaml).unwrap();
    let config = AgentYamlConfig::from_file(&config_path).unwrap();

    let provider = Arc::new(RecordingProvider::new());
    let requested = vec!["pdf_processing".to_string()];
    let agent = mofa_sdk::llm::LLMAgentBuilder::from_yaml_config(config)
        .unwrap()
        .with_provider(provider.clone())
        .try_build()
        .unwrap();

    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let system_prompt = system_text(&provider.take_request().await);
    assert!(system_prompt.contains("# PDF"));

    restore_env_var("XDG_DATA_HOME", previous_data);
    restore_env_var("XDG_CACHE_HOME", previous_cache);
    restore_env_var("XDG_CONFIG_HOME", previous_config);

    server.abort();
}
