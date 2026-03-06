//! 标准 LLM Agent 实现
//!
//! 框架提供的开箱即用的 LLM Agent，用户只需配置 provider 即可使用
//!
//! # 示例
//!
//! ```rust,ignore
//! use mofa_sdk::kernel::AgentInput;
//! use mofa_sdk::runtime::run_agents;
//! use mofa_sdk::llm::LLMAgentBuilder;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let agent = LLMAgentBuilder::from_env()?
//!         .with_id("my-llm-agent")
//!         .with_system_prompt("You are a helpful assistant.")
//!         .build();
//!
//!     let outputs = run_agents(agent, vec![AgentInput::text("Hello")]).await?;
//!     println!("{}", outputs[0].to_text());
//!     Ok(())
//! }
//! ```

use super::client::{ChatSession, LLMClient};
use super::context::{AgentContextBuilder, AgentIdentity, SkillsManager as PromptSkillsManager};
use super::provider::{ChatStream, LLMProvider};
use super::tool_executor::ToolExecutor;
use super::types::{
    ChatCompletionResponse, ChatMessage, LLMError, LLMResult, MessageContent, Role, Tool,
};
use crate::llm::{
    AnthropicConfig, AnthropicProvider, GeminiConfig, GeminiProvider, OllamaConfig, OllamaProvider,
};
use crate::prompt;
use futures::{Stream, StreamExt};
use mofa_kernel::agent::AgentMetadata;
use mofa_kernel::agent::AgentState;
use mofa_kernel::plugin::{AgentPlugin, PluginType};
use mofa_plugins::tts::TTSPlugin;
use std::collections::HashMap;
use std::io::Write;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};

/// Type alias for TTS audio stream - boxed to avoid exposing kokoro-tts types
pub type TtsAudioStream = Pin<Box<dyn Stream<Item = (Vec<f32>, Duration)> + Send>>;

/// Cancellation token for cooperative cancellation
struct CancellationToken {
    cancel: Arc<AtomicBool>,
}

impl CancellationToken {
    fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn clone_token(&self) -> CancellationToken {
        CancellationToken {
            cancel: Arc::clone(&self.cancel),
        }
    }
}

/// 流式文本响应类型
///
/// 每次 yield 一个文本片段（delta content）
pub type TextStream = Pin<Box<dyn Stream<Item = LLMResult<String>> + Send>>;

const HUB_CONFIG_KEY_CATALOG_URL: &str = "skills.hub.catalog_url";
const HUB_CONFIG_KEY_AUTO_INSTALL: &str = "skills.hub.auto_install";
const HUB_CONFIG_KEY_COMPATIBILITY_TARGETS: &str = "skills.hub.compatibility_targets";

/// TTS 流句柄：持有 sink 和消费者任务
///
/// 用于实时流式 TTS，允许 incremental 提交文本
#[cfg(feature = "kokoro")]
struct TTSStreamHandle {
    sink: mofa_plugins::tts::kokoro_wrapper::SynthSink<String>,
    _stream_handle: tokio::task::JoinHandle<()>,
}

/// Active TTS session with cancellation support
struct TTSSession {
    cancellation_token: CancellationToken,
    is_active: Arc<AtomicBool>,
}

impl TTSSession {
    fn new(token: CancellationToken) -> Self {
        let is_active = Arc::new(AtomicBool::new(true));
        TTSSession {
            cancellation_token: token,
            is_active,
        }
    }

    fn cancel(&self) {
        self.cancellation_token.cancel();
        self.is_active.store(false, Ordering::Relaxed);
    }

    fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
}

/// 句子缓冲区：按标点符号断句（内部实现）
struct SentenceBuffer {
    buffer: String,
}

impl SentenceBuffer {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// 推入文本块，返回完整句子（如果有）
    fn push(&mut self, text: &str) -> Option<String> {
        for ch in text.chars() {
            self.buffer.push(ch);
            // 句末标点：。！？!?
            if matches!(ch, '。' | '！' | '？' | '!' | '?') {
                let sentence = self.buffer.trim().to_string();
                if !sentence.is_empty() {
                    self.buffer.clear();
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// 刷新剩余内容
    fn flush(&mut self) -> Option<String> {
        if self.buffer.trim().is_empty() {
            None
        } else {
            let remaining = self.buffer.trim().to_string();
            self.buffer.clear();
            Some(remaining)
        }
    }
}

/// 流式响应事件
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// 文本片段
    Text(String),
    /// 工具调用开始
    ToolCallStart { id: String, name: String },
    /// 工具调用参数片段
    ToolCallDelta { id: String, arguments_delta: String },
    /// 完成原因
    Done(Option<String>),
}

/// LLM Agent 配置
#[derive(Clone)]
pub struct LLMAgentConfig {
    /// Agent ID
    pub agent_id: String,
    /// Agent 名称
    pub name: String,
    /// 系统提示词
    pub system_prompt: Option<String>,
    /// 默认温度
    pub temperature: Option<f32>,
    /// 默认最大 token 数
    pub max_tokens: Option<u32>,
    /// 自定义配置
    pub custom_config: HashMap<String, String>,
    /// 用户 ID，用于数据库持久化和多用户场景
    pub user_id: Option<String>,
    /// 租户 ID，用于多租户支持
    pub tenant_id: Option<String>,
    /// 上下文窗口大小，用于滑动窗口消息管理（单位：轮数/rounds）
    ///
    /// 注意：单位是**轮数**（rounds），不是 token 数量
    /// 每轮对话 ≈ 1 个用户消息 + 1 个助手响应
    pub context_window_size: Option<usize>,
}

impl Default for LLMAgentConfig {
    fn default() -> Self {
        Self {
            agent_id: "llm-agent".to_string(),
            name: "LLM Agent".to_string(),
            system_prompt: None,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            custom_config: HashMap::new(),
            user_id: None,
            tenant_id: None,
            context_window_size: None,
        }
    }
}

/// 标准 LLM Agent
///
/// 框架提供的开箱即用的 LLM Agent 实现
///
/// # 多会话支持
///
/// LLMAgent 支持多会话管理，每个会话有唯一的 session_id：
///
/// ```rust,ignore
/// // 创建新会话
/// let session_id = agent.create_session().await;
///
/// // 使用指定会话对话
/// agent.chat_with_session(&session_id, "Hello").await?;
///
/// // 切换默认会话
/// agent.switch_session(&session_id).await?;
///
/// // 获取所有会话ID
/// let sessions = agent.list_sessions().await;
/// ```
///
/// # TTS 支持
///
/// LLMAgent 支持通过统一的插件系统配置 TTS：
///
/// ```rust,ignore
/// // 创建 TTS 插件（引擎 + 可选音色）
/// let tts_plugin = TTSPlugin::with_engine("tts", kokoro_engine, Some("zf_090"));
///
/// // 通过插件系统添加
/// let agent = LLMAgentBuilder::new()
///     .with_id("my-agent")
///     .with_provider(Arc::new(openai_from_env()?))
///     .with_plugin(tts_plugin)
///     .build();
///
/// // 直接使用 TTS
/// agent.tts_speak("Hello world").await?;
///
/// // 高级用法：自定义配置
/// let tts_plugin = TTSPlugin::with_engine("tts", kokoro_engine, Some("zf_090"))
///     .with_config(TTSPluginConfig {
///         streaming_chunk_size: 8192,
///         ..Default::default()
///     });
/// ```
pub struct LLMAgent {
    config: LLMAgentConfig,
    /// 智能体元数据
    metadata: AgentMetadata,
    client: LLMClient,
    /// 多会话存储 (session_id -> ChatSession)
    sessions: Arc<RwLock<HashMap<String, Arc<RwLock<ChatSession>>>>>,
    /// 当前活动会话ID
    active_session_id: Arc<RwLock<String>>,
    tools: Vec<Tool>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    event_handler: Option<Box<dyn LLMAgentEventHandler>>,
    /// 插件列表
    plugins: Vec<Box<dyn AgentPlugin>>,
    /// 当前智能体状态
    state: AgentState,
    /// 保存 provider 用于创建新会话
    provider: Arc<dyn LLMProvider>,
    /// Prompt 模板插件
    prompt_plugin: Option<Box<dyn prompt::PromptTemplatePlugin>>,
    /// TTS 插件（通过 builder 配置）
    tts_plugin: Option<Arc<Mutex<TTSPlugin>>>,
    /// Optional prompt context builder for bootstrap files and skills.
    context_builder: Option<AgentContextBuilder>,
    /// 缓存的 Kokoro TTS 引擎（只需初始化一次，后续可复用）
    #[cfg(feature = "kokoro")]
    cached_kokoro_engine: Arc<Mutex<Option<Arc<mofa_plugins::tts::kokoro_wrapper::KokoroTTS>>>>,
    /// Active TTS session for cancellation
    active_tts_session: Arc<Mutex<Option<TTSSession>>>,
    /// 持久化存储（可选，用于从数据库加载历史会话）
    message_store: Option<Arc<dyn crate::persistence::MessageStore + Send + Sync>>,
    session_store: Option<Arc<dyn crate::persistence::SessionStore + Send + Sync>>,
    /// 用户 ID（用于从数据库加载会话）
    persistence_user_id: Option<uuid::Uuid>,
    /// Agent ID（用于从数据库加载会话）
    persistence_agent_id: Option<uuid::Uuid>,
}

/// LLM Agent 事件处理器
///
/// 允许用户自定义事件处理逻辑
#[async_trait::async_trait]
pub trait LLMAgentEventHandler: Send + Sync {
    /// Clone this handler trait object
    fn clone_box(&self) -> Box<dyn LLMAgentEventHandler>;

    /// 获取 Any 类型用于 downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// 处理用户消息前的钩子
    async fn before_chat(&self, message: &str) -> LLMResult<Option<String>> {
        Ok(Some(message.to_string()))
    }

    /// 处理用户消息前的钩子（带模型名称）
    ///
    /// 默认实现调用 `before_chat`。
    /// 如果需要知道使用的模型名称（例如用于持久化），请实现此方法。
    async fn before_chat_with_model(
        &self,
        message: &str,
        _model: &str,
    ) -> LLMResult<Option<String>> {
        self.before_chat(message).await
    }

    /// 处理 LLM 响应后的钩子
    async fn after_chat(&self, response: &str) -> LLMResult<Option<String>> {
        Ok(Some(response.to_string()))
    }

    /// 处理 LLM 响应后的钩子（带元数据）
    ///
    /// 默认实现调用 after_chat。
    /// 如果需要访问响应元数据（如 response_id, model, token counts），请实现此方法。
    async fn after_chat_with_metadata(
        &self,
        response: &str,
        _metadata: &super::types::LLMResponseMetadata,
    ) -> LLMResult<Option<String>> {
        self.after_chat(response).await
    }

    /// 处理工具调用
    async fn on_tool_call(&self, name: &str, arguments: &str) -> LLMResult<Option<String>> {
        let _ = (name, arguments);
        Ok(None)
    }

    /// 处理错误
    async fn on_error(&self, error: &LLMError) -> LLMResult<Option<String>> {
        let _ = error;
        Ok(None)
    }
}

impl Clone for Box<dyn LLMAgentEventHandler> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl LLMAgent {
    /// 创建新的 LLM Agent
    pub fn new(config: LLMAgentConfig, provider: Arc<dyn LLMProvider>) -> Self {
        Self::with_initial_session(config, provider, None)
    }

    /// 创建新的 LLM Agent，并指定初始会话 ID
    ///
    /// # 参数
    /// - `config`: Agent 配置
    /// - `provider`: LLM Provider
    /// - `initial_session_id`: 初始会话 ID，如果为 None 则使用自动生成的 ID
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgent::with_initial_session(
    ///     config,
    ///     provider,
    ///     Some("user-session-001".to_string())
    /// );
    /// ```
    pub fn with_initial_session(
        config: LLMAgentConfig,
        provider: Arc<dyn LLMProvider>,
        initial_session_id: Option<String>,
    ) -> Self {
        let client = LLMClient::new(provider.clone());

        let mut session = if let Some(sid) = initial_session_id {
            ChatSession::with_id_str(&sid, LLMClient::new(provider.clone()))
        } else {
            ChatSession::new(LLMClient::new(provider.clone()))
        };

        // 设置系统提示
        if let Some(ref prompt) = config.system_prompt {
            session = session.with_system(prompt.clone());
        }

        // 设置上下文窗口大小
        session = session.with_context_window_size(config.context_window_size);

        let session_id = session.session_id().to_string();
        let session_arc = Arc::new(RwLock::new(session));

        // 初始化会话存储
        let mut sessions = HashMap::new();
        sessions.insert(session_id.clone(), session_arc);

        // Clone fields needed for metadata before moving config
        let agent_id = config.agent_id.clone();
        let name = config.name.clone();

        // 创建 AgentCapabilities
        let capabilities = mofa_kernel::agent::AgentCapabilities::builder()
            .tags(vec![
                "llm".to_string(),
                "chat".to_string(),
                "text-generation".to_string(),
                "multi-session".to_string(),
            ])
            .build();

        Self {
            config,
            metadata: AgentMetadata {
                id: agent_id,
                name,
                description: None,
                version: None,
                capabilities,
                state: AgentState::Created,
            },
            client,
            sessions: Arc::new(RwLock::new(sessions)),
            active_session_id: Arc::new(RwLock::new(session_id)),
            tools: Vec::new(),
            tool_executor: None,
            event_handler: None,
            plugins: Vec::new(),
            state: AgentState::Created,
            provider,
            prompt_plugin: None,
            tts_plugin: None,
            context_builder: None,
            #[cfg(feature = "kokoro")]
            cached_kokoro_engine: Arc::new(Mutex::new(None)),
            active_tts_session: Arc::new(Mutex::new(None)),
            message_store: None,
            session_store: None,
            persistence_user_id: None,
            persistence_agent_id: None,
        }
    }

    /// 创建新的 LLM Agent，并尝试从数据库加载初始会话（异步版本）
    ///
    /// 如果提供了 persistence stores 且 session_id 存在于数据库中，
    /// 会自动加载历史消息并应用滑动窗口。
    ///
    /// # 参数
    /// - `config`: Agent 配置
    /// - `provider`: LLM Provider
    /// - `initial_session_id`: 初始会话 ID，如果为 None 则使用自动生成的 ID
    /// - `message_store`: 消息存储（可选，用于从数据库加载历史）
    /// - `session_store`: 会话存储（可选，用于从数据库加载历史）
    /// - `persistence_user_id`: 用户 ID（用于从数据库加载会话）
    /// - `persistence_agent_id`: Agent ID（用于从数据库加载会话）
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgent::with_initial_session_async(
    ///     config,
    ///     provider,
    ///     Some("user-session-001".to_string()),
    ///     Some(message_store),
    ///     Some(session_store),
    ///     Some(user_id),
    ///     Some(agent_id),
    /// ).await?;
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub async fn with_initial_session_async(
        config: LLMAgentConfig,
        provider: Arc<dyn LLMProvider>,
        initial_session_id: Option<String>,
        message_store: Option<Arc<dyn crate::persistence::MessageStore + Send + Sync>>,
        session_store: Option<Arc<dyn crate::persistence::SessionStore + Send + Sync>>,
        persistence_user_id: Option<uuid::Uuid>,
        persistence_tenant_id: Option<uuid::Uuid>,
        persistence_agent_id: Option<uuid::Uuid>,
    ) -> Self {
        let client = LLMClient::new(provider.clone());

        // Clone initial_session_id to avoid move issues
        let initial_session_id_clone = initial_session_id.clone();

        // 1. 尝试从数据库加载会话（如果有 stores 且指定了 session_id）
        let session = if let (
            Some(sid),
            Some(msg_store),
            Some(sess_store),
            Some(user_id),
            Some(tenant_id),
            Some(agent_id),
        ) = (
            initial_session_id_clone,
            message_store.clone(),
            session_store.clone(),
            persistence_user_id,
            persistence_tenant_id,
            persistence_agent_id,
        ) {
            // Clone stores before moving them into ChatSession::load
            let msg_store_clone = msg_store.clone();
            let sess_store_clone = sess_store.clone();

            let session_uuid = uuid::Uuid::parse_str(&sid).unwrap_or_else(|_| {
                tracing::warn!("⚠️ 无效的 session_id 格式 '{}', 将生成新的 UUID", sid);
                uuid::Uuid::now_v7()
            });

            // 尝试从数据库加载
            match ChatSession::load(
                session_uuid,
                LLMClient::new(provider.clone()),
                user_id,
                agent_id,
                tenant_id,
                msg_store,
                sess_store,
                config.context_window_size,
            )
            .await
            {
                Ok(loaded_session) => {
                    tracing::info!(
                        "✅ 从数据库加载会话: {} ({} 条消息)",
                        sid,
                        loaded_session.messages().len()
                    );
                    loaded_session
                }
                Err(e) => {
                    // 会话不存在，创建新会话（使用用户指定的ID和从persistence获取的user_id/agent_id）
                    tracing::info!("📝 创建新会话并持久化: {} (数据库中不存在: {})", sid, e);

                    // Clone stores again for the fallback case
                    let msg_store_clone2 = msg_store_clone.clone();
                    let sess_store_clone2 = sess_store_clone.clone();

                    // 使用正确的 user_id 和 agent_id 创建会话，并持久化到数据库
                    match ChatSession::with_id_and_stores_and_persist(
                        session_uuid,
                        LLMClient::new(provider.clone()),
                        user_id,
                        agent_id,
                        tenant_id,
                        msg_store_clone,
                        sess_store_clone,
                        config.context_window_size,
                    )
                    .await
                    {
                        Ok(mut new_session) => {
                            if let Some(ref prompt) = config.system_prompt {
                                new_session = new_session.with_system(prompt.clone());
                            }
                            new_session
                        }
                        Err(persist_err) => {
                            tracing::error!("❌ 持久化会话失败: {}, 降级为内存会话", persist_err);
                            // 降级：如果持久化失败，创建内存会话
                            let new_session = ChatSession::with_id_and_stores(
                                session_uuid,
                                LLMClient::new(provider.clone()),
                                user_id,
                                agent_id,
                                tenant_id,
                                msg_store_clone2,
                                sess_store_clone2,
                                config.context_window_size,
                            );
                            if let Some(ref prompt) = config.system_prompt {
                                new_session.with_system(prompt.clone())
                            } else {
                                new_session
                            }
                        }
                    }
                }
            }
        } else {
            // 没有 persistence stores，创建普通会话
            let mut session = if let Some(sid) = initial_session_id {
                ChatSession::with_id_str(&sid, LLMClient::new(provider.clone()))
            } else {
                ChatSession::new(LLMClient::new(provider.clone()))
            };
            if let Some(ref prompt) = config.system_prompt {
                session = session.with_system(prompt.clone());
            }
            session.with_context_window_size(config.context_window_size)
        };

        let session_id = session.session_id().to_string();
        let session_arc = Arc::new(RwLock::new(session));

        // 初始化会话存储
        let mut sessions = HashMap::new();
        sessions.insert(session_id.clone(), session_arc);

        // Clone fields needed for metadata before moving config
        let agent_id = config.agent_id.clone();
        let name = config.name.clone();

        // 创建 AgentCapabilities
        let capabilities = mofa_kernel::agent::AgentCapabilities::builder()
            .tags(vec![
                "llm".to_string(),
                "chat".to_string(),
                "text-generation".to_string(),
                "multi-session".to_string(),
            ])
            .build();

        Self {
            config,
            metadata: AgentMetadata {
                id: agent_id,
                name,
                description: None,
                version: None,
                capabilities,
                state: AgentState::Created,
            },
            client,
            sessions: Arc::new(RwLock::new(sessions)),
            active_session_id: Arc::new(RwLock::new(session_id)),
            tools: Vec::new(),
            tool_executor: None,
            event_handler: None,
            plugins: Vec::new(),
            state: AgentState::Created,
            provider,
            prompt_plugin: None,
            tts_plugin: None,
            context_builder: None,
            #[cfg(feature = "kokoro")]
            cached_kokoro_engine: Arc::new(Mutex::new(None)),
            active_tts_session: Arc::new(Mutex::new(None)),
            message_store,
            session_store,
            persistence_user_id,
            persistence_agent_id,
        }
    }

    /// 获取配置
    pub fn config(&self) -> &LLMAgentConfig {
        &self.config
    }

    /// 获取 LLM Client
    pub fn client(&self) -> &LLMClient {
        &self.client
    }

    // ========================================================================
    // 会话管理方法
    // ========================================================================

    /// 获取当前活动会话ID
    pub async fn current_session_id(&self) -> String {
        self.active_session_id.read().await.clone()
    }

    async fn resolve_dynamic_system_prompt(&self) -> Option<String> {
        let mut system_prompt = self.config.system_prompt.clone();

        if let Some(ref plugin) = self.prompt_plugin
            && let Some(template) = plugin.get_current_template().await
        {
            system_prompt = match template.render(&[]) {
                Ok(prompt) => Some(prompt),
                Err(_) => self.config.system_prompt.clone(),
            };
        }

        system_prompt
    }

    async fn resolve_session_system_prompt(&self) -> LLMResult<Option<String>> {
        let base_prompt = self.resolve_dynamic_system_prompt().await;

        if let Some(context_builder) = &self.context_builder {
            context_builder.clear_cache().await;
            let context_prompt = context_builder
                .build_system_prompt()
                .await
                .map_err(|e| LLMError::Other(e.to_string()))?;
            return Ok(Some(match base_prompt {
                Some(base_prompt) if !base_prompt.trim().is_empty() => {
                    format!("{}\n\n---\n\n{}", base_prompt, context_prompt)
                }
                _ => context_prompt,
            }));
        }

        Ok(base_prompt)
    }

    async fn build_context_messages(
        &self,
        history: Vec<ChatMessage>,
        current: &str,
        requested_skills: Option<&[String]>,
    ) -> LLMResult<Vec<ChatMessage>> {
        let context_builder = self
            .context_builder
            .as_ref()
            .ok_or_else(|| LLMError::Other("context builder is not configured".to_string()))?;

        context_builder.clear_cache().await;
        let mut messages = if let Some(names) = requested_skills {
            context_builder
                .build_messages_with_skills(history, current, None, Some(names))
                .await
                .map_err(|e| LLMError::Other(e.to_string()))?
        } else {
            context_builder
                .build_messages(history, current, None)
                .await
                .map_err(|e| LLMError::Other(e.to_string()))?
        };

        if let Some(base_prompt) = self.resolve_dynamic_system_prompt().await {
            Self::prepend_system_prompt(&mut messages, &base_prompt);
        }

        Ok(messages)
    }

    fn prepend_system_prompt(messages: &mut Vec<ChatMessage>, prompt: &str) {
        if prompt.trim().is_empty() {
            return;
        }

        if let Some(first) = messages.first_mut()
            && first.role == Role::System
            && let Some(MessageContent::Text(content)) = &mut first.content
        {
            let merged = format!("{}\n\n---\n\n{}", prompt, content);
            *content = merged;
            return;
        }

        messages.insert(0, ChatMessage::system(prompt));
    }

    async fn send_messages(&self, messages: Vec<ChatMessage>) -> LLMResult<ChatCompletionResponse> {
        let mut builder = self.client.chat().messages(messages);

        if let Some(temp) = self.config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(tokens) = self.config.max_tokens {
            builder = builder.max_tokens(tokens);
        }

        if let Some(ref executor) = self.tool_executor {
            let tools = if self.tools.is_empty() {
                executor.available_tools().await?
            } else {
                self.tools.clone()
            };

            if !tools.is_empty() {
                builder = builder.tools(tools);
            }

            builder = builder.with_tool_executor(executor.clone());
            return builder.send_with_tools().await;
        }

        builder.send().await
    }

    async fn send_stream_messages(&self, messages: Vec<ChatMessage>) -> LLMResult<ChatStream> {
        let mut builder = self.client.chat().messages(messages);

        if let Some(temp) = self.config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(tokens) = self.config.max_tokens {
            builder = builder.max_tokens(tokens);
        }

        builder.send_stream().await
    }

    fn push_session_messages(
        session: &mut ChatSession,
        user_message: &str,
        assistant_message: &str,
        metadata: super::types::LLMResponseMetadata,
    ) {
        session.messages_mut().push(ChatMessage::user(user_message));
        session
            .messages_mut()
            .push(ChatMessage::assistant(assistant_message));

        if session.context_window_size().is_some() {
            let current_messages = session.messages().to_vec();
            *session.messages_mut() = ChatSession::apply_sliding_window_static(
                &current_messages,
                session.context_window_size(),
            );
        }

        session.set_last_response_metadata(metadata);
    }

    /// 创建新会话
    ///
    /// 返回新会话的 session_id
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let session_id = agent.create_session().await;
    /// agent.chat_with_session(&session_id, "Hello").await?;
    /// ```
    pub async fn create_session(&self) -> String {
        let mut session = ChatSession::new(LLMClient::new(self.provider.clone()));

        if let Ok(Some(prompt)) = self.resolve_session_system_prompt().await {
            session = session.with_system(prompt.clone());
        }

        // 设置上下文窗口大小
        session = session.with_context_window_size(self.config.context_window_size);

        let session_id = session.session_id().to_string();
        let session_arc = Arc::new(RwLock::new(session));

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session_arc);

        session_id
    }

    /// 使用指定ID创建新会话
    ///
    /// 如果 session_id 已存在，返回错误
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let session_id = agent.create_session_with_id("user-123-session").await?;
    /// ```
    pub async fn create_session_with_id(&self, session_id: impl Into<String>) -> LLMResult<String> {
        let session_id = session_id.into();

        {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(&session_id) {
                return Err(LLMError::Other(format!(
                    "Session with id '{}' already exists",
                    session_id
                )));
            }
        }

        let mut session =
            ChatSession::with_id_str(&session_id, LLMClient::new(self.provider.clone()));

        if let Some(prompt) = self.resolve_session_system_prompt().await? {
            session = session.with_system(prompt.clone());
        }

        // 设置上下文窗口大小
        session = session.with_context_window_size(self.config.context_window_size);

        let session_arc = Arc::new(RwLock::new(session));

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session_arc);

        Ok(session_id)
    }

    /// 切换当前活动会话
    ///
    /// # 错误
    /// 如果 session_id 不存在则返回错误
    pub async fn switch_session(&self, session_id: &str) -> LLMResult<()> {
        let sessions = self.sessions.read().await;
        if !sessions.contains_key(session_id) {
            return Err(LLMError::Other(format!(
                "Session '{}' not found",
                session_id
            )));
        }
        drop(sessions);

        let mut active = self.active_session_id.write().await;
        *active = session_id.to_string();
        Ok(())
    }

    /// 获取或创建会话
    ///
    /// 如果 session_id 存在则返回它，否则使用该 ID 创建新会话
    pub async fn get_or_create_session(&self, session_id: impl Into<String>) -> String {
        let session_id = session_id.into();

        {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(&session_id) {
                return session_id;
            }
        }

        // 会话不存在，创建新的
        let _ = self.create_session_with_id(&session_id).await;
        session_id
    }

    /// 删除会话
    ///
    /// # 注意
    /// 不能删除当前活动会话，需要先切换到其他会话
    pub async fn remove_session(&self, session_id: &str) -> LLMResult<()> {
        let active = self.active_session_id.read().await.clone();
        if active == session_id {
            return Err(LLMError::Other(
                "Cannot remove active session. Switch to another session first.".to_string(),
            ));
        }

        let mut sessions = self.sessions.write().await;
        if sessions.remove(session_id).is_none() {
            return Err(LLMError::Other(format!(
                "Session '{}' not found",
                session_id
            )));
        }

        Ok(())
    }

    /// 列出所有会话ID
    pub async fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.read().await;
        sessions.keys().cloned().collect()
    }

    /// 获取会话数量
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    /// 检查会话是否存在
    pub async fn has_session(&self, session_id: &str) -> bool {
        let sessions = self.sessions.read().await;
        sessions.contains_key(session_id)
    }

    // ========================================================================
    // TTS 便捷方法
    // ========================================================================

    /// 使用 TTS 合成并播放文本
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// agent.tts_speak("Hello world").await?;
    /// ```
    pub async fn tts_speak(&self, text: &str) -> LLMResult<()> {
        let tts = self
            .tts_plugin
            .as_ref()
            .ok_or_else(|| LLMError::Other("TTS plugin not configured".to_string()))?;

        let mut tts_guard = tts.lock().await;
        tts_guard
            .synthesize_and_play(text)
            .await
            .map_err(|e| LLMError::Other(format!("TTS synthesis failed: {}", e)))
    }

    /// 使用 TTS 流式合成文本
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// agent.tts_speak_streaming("Hello world", Box::new(|audio| {
    ///     println!("Got {} bytes of audio", audio.len());
    /// })).await?;
    /// ```
    pub async fn tts_speak_streaming(
        &self,
        text: &str,
        callback: Box<dyn Fn(Vec<u8>) + Send + Sync>,
    ) -> LLMResult<()> {
        let tts = self
            .tts_plugin
            .as_ref()
            .ok_or_else(|| LLMError::Other("TTS plugin not configured".to_string()))?;

        let mut tts_guard = tts.lock().await;
        tts_guard
            .synthesize_streaming(text, callback)
            .await
            .map_err(|e| LLMError::Other(format!("TTS streaming failed: {}", e)))
    }

    /// 使用 TTS 流式合成文本（f32 native format，更高效）
    ///
    /// This method is more efficient for KokoroTTS as it uses the native f32 format
    /// without the overhead of f32 -> i16 -> u8 conversion.
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use rodio::buffer::SamplesBuffer;
    ///
    /// agent.tts_speak_f32_stream("Hello world", Box::new(|audio_f32| {
    ///     // audio_f32 is Vec<f32> with values in [-1.0, 1.0]
    ///     sink.append(SamplesBuffer::new(1, 24000, audio_f32));
    /// })).await?;
    /// ```
    pub async fn tts_speak_f32_stream(
        &self,
        text: &str,
        callback: Box<dyn Fn(Vec<f32>) + Send + Sync>,
    ) -> LLMResult<()> {
        let tts = self
            .tts_plugin
            .as_ref()
            .ok_or_else(|| LLMError::Other("TTS plugin not configured".to_string()))?;

        let mut tts_guard = tts.lock().await;
        tts_guard
            .synthesize_streaming_f32(text, callback)
            .await
            .map_err(|e| LLMError::Other(format!("TTS f32 streaming failed: {}", e)))
    }

    /// 获取 TTS 音频流（仅支持 Kokoro TTS）
    ///
    /// Returns a direct stream of (audio_f32, duration) tuples from KokoroTTS.
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    /// use rodio::buffer::SamplesBuffer;
    ///
    /// if let Ok(mut stream) = agent.tts_create_stream("Hello world").await {
    ///     while let Some((audio, took)) = stream.next().await {
    ///         // audio is Vec<f32> with values in [-1.0, 1.0]
    ///         sink.append(SamplesBuffer::new(1, 24000, audio));
    ///     }
    /// }
    /// ```
    pub async fn tts_create_stream(&self, text: &str) -> LLMResult<TtsAudioStream> {
        #[cfg(feature = "kokoro")]
        {
            use mofa_plugins::tts::kokoro_wrapper::KokoroTTS;

            // 首先检查是否有缓存的引擎（只需初始化一次）
            let cached_engine = {
                let cache_guard = self.cached_kokoro_engine.lock().await;
                cache_guard.clone()
            };

            let kokoro = if let Some(engine) = cached_engine {
                // 使用缓存的引擎（无需再次获取 tts_plugin 的锁）
                engine
            } else {
                // 首次调用：获取 tts_plugin 的锁，downcast 并缓存
                let tts = self
                    .tts_plugin
                    .as_ref()
                    .ok_or_else(|| LLMError::Other("TTS plugin not configured".to_string()))?;

                let tts_guard = tts.lock().await;

                let engine = tts_guard
                    .engine()
                    .ok_or_else(|| LLMError::Other("TTS engine not initialized".to_string()))?;

                if let Some(kokoro_ref) = engine.as_any().downcast_ref::<KokoroTTS>() {
                    // 克隆 KokoroTTS（内部使用 Arc，克隆只是增加引用计数）
                    let cloned = kokoro_ref.clone();
                    let cloned_arc = Arc::new(cloned);

                    // 获取 voice 配置
                    let voice = tts_guard
                        .stats()
                        .get("default_voice")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default");

                    // 缓存克隆的引擎
                    {
                        let mut cache_guard = self.cached_kokoro_engine.lock().await;
                        *cache_guard = Some(cloned_arc.clone());
                    }

                    cloned_arc
                } else {
                    return Err(LLMError::Other("TTS engine is not KokoroTTS".to_string()));
                }
            };

            // 使用缓存的引擎创建 stream（无需再次获取 tts_plugin 的锁）
            let voice = "default"; // 可以从配置中获取
            let (mut sink, stream) = kokoro
                .create_stream(voice)
                .await
                .map_err(|e| LLMError::Other(format!("Failed to create TTS stream: {}", e)))?;

            // Submit text for synthesis
            sink.synth(text.to_string()).await.map_err(|e| {
                LLMError::Other(format!("Failed to submit text for synthesis: {}", e))
            })?;

            // Box the stream to hide the concrete type
            Ok(Box::pin(stream))
        }

        #[cfg(not(feature = "kokoro"))]
        {
            Err(LLMError::Other("Kokoro feature not enabled".to_string()))
        }
    }

    /// Stream multiple sentences through a single TTS stream
    ///
    /// This is more efficient than calling tts_speak_f32_stream multiple times
    /// because it reuses the same stream for all sentences, following the kokoro-tts
    /// streaming pattern: ONE stream, multiple synth calls, continuous audio output.
    ///
    /// # Arguments
    /// - `sentences`: Vector of text sentences to synthesize
    /// - `callback`: Function to call with each audio chunk (Vec<f32>)
    ///
    /// # Example
    /// ```rust,ignore
    /// use rodio::buffer::SamplesBuffer;
    ///
    /// let sentences = vec!["Hello".to_string(), "World".to_string()];
    /// agent.tts_speak_f32_stream_batch(
    ///     sentences,
    ///     Box::new(|audio_f32| {
    ///         sink.append(SamplesBuffer::new(1, 24000, audio_f32));
    ///     }),
    /// ).await?;
    /// ```
    pub async fn tts_speak_f32_stream_batch(
        &self,
        sentences: Vec<String>,
        callback: Box<dyn Fn(Vec<f32>) + Send + Sync>,
    ) -> LLMResult<()> {
        let tts = self
            .tts_plugin
            .as_ref()
            .ok_or_else(|| LLMError::Other("TTS plugin not configured".to_string()))?;

        let tts_guard = tts.lock().await;

        #[cfg(feature = "kokoro")]
        {
            use mofa_plugins::tts::kokoro_wrapper::KokoroTTS;

            let engine = tts_guard
                .engine()
                .ok_or_else(|| LLMError::Other("TTS engine not initialized".to_string()))?;

            if let Some(kokoro) = engine.as_any().downcast_ref::<KokoroTTS>() {
                let voice = tts_guard
                    .stats()
                    .get("default_voice")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();

                // Create ONE stream for all sentences
                let (mut sink, mut stream) = kokoro
                    .create_stream(&voice)
                    .await
                    .map_err(|e| LLMError::Other(format!("Failed to create TTS stream: {}", e)))?;

                // Spawn a task to consume the stream continuously
                tokio::spawn(async move {
                    while let Some((audio, _took)) = stream.next().await {
                        callback(audio);
                    }
                });

                // Submit all sentences to the same sink
                for sentence in sentences {
                    sink.synth(sentence)
                        .await
                        .map_err(|e| LLMError::Other(format!("Failed to submit text: {}", e)))?;
                }

                return Ok(());
            }

            Err(LLMError::Other("TTS engine is not KokoroTTS".to_string()))
        }

        #[cfg(not(feature = "kokoro"))]
        {
            Err(LLMError::Other("Kokoro feature not enabled".to_string()))
        }
    }

    /// 检查是否配置了 TTS 插件
    pub fn has_tts(&self) -> bool {
        self.tts_plugin.is_some()
    }

    /// Interrupt currently playing TTS audio
    ///
    /// Stops current audio playback and cancels any ongoing TTS synthesis.
    /// Call this before starting a new TTS request for clean transition.
    ///
    /// # Example
    /// ```rust,ignore
    /// // User enters new input while audio is playing
    /// agent.interrupt_tts().await?;
    /// agent.chat_with_tts(&session_id, new_input).await?;
    /// ```
    pub async fn interrupt_tts(&self) -> LLMResult<()> {
        let mut session_guard = self.active_tts_session.lock().await;
        if let Some(session) = session_guard.take() {
            session.cancel();
        }
        Ok(())
    }

    // ========================================================================
    // LLM + TTS 流式对话方法
    // ========================================================================

    /// 流式聊天并自动 TTS 播放（最简版本）
    ///
    /// 自动处理：
    /// - 流式 LLM 输出
    /// - 按标点断句
    /// - 批量 TTS 播放
    ///
    /// # 示例
    /// ```rust,ignore
    /// agent.chat_with_tts(&session_id, "你好").await?;
    /// ```
    pub async fn chat_with_tts(
        &self,
        session_id: &str,
        message: impl Into<String>,
    ) -> LLMResult<()> {
        self.chat_with_tts_internal(session_id, message, None).await
    }

    /// 流式聊天并自动 TTS 播放（自定义音频处理）
    ///
    /// # 示例
    /// ```rust,ignore
    /// use rodio::buffer::SamplesBuffer;
    ///
    /// agent.chat_with_tts_callback(&session_id, "你好", |audio| {
    ///     sink.append(SamplesBuffer::new(1, 24000, audio));
    /// }).await?;
    /// ```
    pub async fn chat_with_tts_callback(
        &self,
        session_id: &str,
        message: impl Into<String>,
        callback: impl Fn(Vec<f32>) + Send + Sync + 'static,
    ) -> LLMResult<()> {
        self.chat_with_tts_internal(session_id, message, Some(Box::new(callback)))
            .await
    }

    /// 创建实时 TTS 流
    ///
    /// 返回的 handle 允许 incremental 提交文本，实现真正的实时流式 TTS。
    ///
    /// # 核心机制
    /// 1. 创建 TTS stream（仅一次）
    /// 2. 启动消费者任务（持续接收音频块）
    /// 3. 返回的 sink 支持多次 `synth()` 调用
    #[cfg(feature = "kokoro")]
    async fn create_tts_stream_handle(
        &self,
        callback: Box<dyn Fn(Vec<f32>) + Send + Sync>,
        cancellation_token: Option<CancellationToken>,
    ) -> LLMResult<TTSStreamHandle> {
        use mofa_plugins::tts::kokoro_wrapper::KokoroTTS;

        let tts = self
            .tts_plugin
            .as_ref()
            .ok_or_else(|| LLMError::Other("TTS plugin not configured".to_string()))?;

        let tts_guard = tts.lock().await;
        let engine = tts_guard
            .engine()
            .ok_or_else(|| LLMError::Other("TTS engine not initialized".to_string()))?;

        let kokoro = engine
            .as_any()
            .downcast_ref::<KokoroTTS>()
            .ok_or_else(|| LLMError::Other("TTS engine is not KokoroTTS".to_string()))?;

        let voice = tts_guard
            .stats()
            .get("default_voice")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();

        // 创建 TTS stream（只创建一次）
        let (sink, mut stream) = kokoro
            .create_stream(&voice)
            .await
            .map_err(|e| LLMError::Other(format!("Failed to create TTS stream: {}", e)))?;

        // Clone cancellation token for the spawned task
        let token_clone = cancellation_token.as_ref().map(|t| t.clone_token());

        // 启动消费者任务（持续接收音频块，支持取消检查）
        let stream_handle = tokio::spawn(async move {
            while let Some((audio, _took)) = stream.next().await {
                // 检查取消信号
                if let Some(ref token) = token_clone
                    && token.is_cancelled()
                {
                    break; // 退出循环，停止音频处理
                }
                callback(audio);
            }
        });

        Ok(TTSStreamHandle {
            sink,
            _stream_handle: stream_handle,
        })
    }

    /// 内部实现：LLM + TTS 实时流式对话
    ///
    /// # 核心机制
    /// 1. 在 LLM 流式输出**之前**创建 TTS stream
    /// 2. 检测到完整句子时立即提交到 TTS
    /// 3. LLM 流和 TTS 流并行运行
    async fn chat_with_tts_internal(
        &self,
        session_id: &str,
        message: impl Into<String>,
        callback: Option<Box<dyn Fn(Vec<f32>) + Send + Sync>>,
    ) -> LLMResult<()> {
        #[cfg(feature = "kokoro")]
        {
            use mofa_plugins::tts::kokoro_wrapper::KokoroTTS;

            let callback = match callback {
                Some(cb) => cb,
                None => {
                    // 无 TTS 请求，仅流式输出文本
                    let mut text_stream =
                        self.chat_stream_with_session(session_id, message).await?;
                    while let Some(result) = text_stream.next().await {
                        match result {
                            Ok(text_chunk) => {
                                print!("{}", text_chunk);
                                std::io::stdout().flush().map_err(|e| {
                                    LLMError::Other(format!("Failed to flush stdout: {}", e))
                                })?;
                            }
                            Err(e) if e.to_string().contains("__stream_end__") => break,
                            Err(e) => return Err(e),
                        }
                    }
                    println!();
                    return Ok(());
                }
            };

            // Step 0: 取消任何现有的 TTS 会话
            self.interrupt_tts().await?;

            // Step 1: 创建 cancellation token
            let cancellation_token = CancellationToken::new();

            // Step 2: 在 LLM 流式输出之前创建 TTS stream（传入 cancellation token）
            let mut tts_handle = self
                .create_tts_stream_handle(callback, Some(cancellation_token.clone_token()))
                .await?;

            // Step 3: 创建并跟踪新的 TTS session
            let session = TTSSession::new(cancellation_token);

            {
                let mut active_session = self.active_tts_session.lock().await;
                *active_session = Some(session);
            }

            let mut buffer = SentenceBuffer::new();

            // Step 4: 流式处理 LLM 响应，实时提交句子到 TTS
            let mut text_stream = self.chat_stream_with_session(session_id, message).await?;

            while let Some(result) = text_stream.next().await {
                match result {
                    Ok(text_chunk) => {
                        // 检查是否已被取消
                        {
                            let active_session = self.active_tts_session.lock().await;
                            if let Some(ref session) = *active_session
                                && !session.is_active()
                            {
                                return Ok(()); // 优雅退出
                            }
                        }

                        // 实时显示文本
                        print!("{}", text_chunk);
                        std::io::stdout().flush().map_err(|e| {
                            LLMError::Other(format!("Failed to flush stdout: {}", e))
                        })?;

                        // 检测句子并立即提交到 TTS
                        if let Some(sentence) = buffer.push(&text_chunk)
                            && let Err(e) = tts_handle.sink.synth(sentence).await
                        {
                            eprintln!("[TTS Error] Failed to submit sentence: {}", e);
                            // 继续流式处理，即使 TTS 失败
                        }
                    }
                    Err(e) if e.to_string().contains("__stream_end__") => break,
                    Err(e) => return Err(e),
                }
            }

            // Step 5: 提交剩余文本
            if let Some(remaining) = buffer.flush()
                && let Err(e) = tts_handle.sink.synth(remaining).await
            {
                eprintln!("[TTS Error] Failed to submit final sentence: {}", e);
            }

            // Step 6: 清理会话
            {
                let mut active_session = self.active_tts_session.lock().await;
                *active_session = None;
            }

            // Step 7: 等待 TTS 流完成（所有音频块处理完毕）
            let _ = tokio::time::timeout(
                tokio::time::Duration::from_secs(30),
                tts_handle._stream_handle,
            )
            .await
            .map_err(|_| LLMError::Other("TTS stream processing timeout".to_string()))
            .and_then(|r| r.map_err(|e| LLMError::Other(format!("TTS stream task failed: {}", e))));

            Ok(())
        }

        #[cfg(not(feature = "kokoro"))]
        {
            // 当 kokoro feature 未启用时，使用批量处理模式
            let mut text_stream = self.chat_stream_with_session(session_id, message).await?;
            let mut buffer = SentenceBuffer::new();
            let mut sentences = Vec::new();

            // 收集所有句子
            while let Some(result) = text_stream.next().await {
                match result {
                    Ok(text_chunk) => {
                        print!("{}", text_chunk);
                        std::io::stdout().flush().map_err(|e| {
                            LLMError::Other(format!("Failed to flush stdout: {}", e))
                        })?;

                        if let Some(sentence) = buffer.push(&text_chunk) {
                            sentences.push(sentence);
                        }
                    }
                    Err(e) if e.to_string().contains("__stream_end__") => break,
                    Err(e) => return Err(e),
                }
            }

            // 添加剩余内容
            if let Some(remaining) = buffer.flush() {
                sentences.push(remaining);
            }

            // 批量播放 TTS（如果有回调）
            if !sentences.is_empty()
                && let Some(cb) = callback
            {
                for sentence in &sentences {
                    println!("\n[TTS] {}", sentence);
                }
                // 注意：非 kokoro 环境下无法调用此方法
                // 这里需要根据实际情况处理
                let _ = cb;
            }

            Ok(())
        }
    }

    /// 内部方法：获取会话 Arc
    async fn get_session_arc(&self, session_id: &str) -> LLMResult<Arc<RwLock<ChatSession>>> {
        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| LLMError::Other(format!("Session '{}' not found", session_id)))
    }

    // ========================================================================
    // 对话方法
    // ========================================================================

    /// 发送消息并获取响应（使用当前活动会话）
    pub async fn chat(&self, message: impl Into<String>) -> LLMResult<String> {
        let session_id = self.active_session_id.read().await.clone();
        self.chat_with_session(&session_id, message).await
    }

    /// 使用指定会话发送消息并获取响应
    ///
    /// # 参数
    /// - `session_id`: 会话唯一标识
    /// - `message`: 用户消息
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let session_id = agent.create_session().await;
    /// let response = agent.chat_with_session(&session_id, "Hello").await?;
    /// ```
    pub async fn chat_with_session(
        &self,
        session_id: &str,
        message: impl Into<String>,
    ) -> LLMResult<String> {
        let message = message.into();

        if self.context_builder.is_some() {
            return self
                .chat_with_session_and_skills(session_id, message, &[])
                .await;
        }

        // 获取模型名称
        let model = self.provider.default_model();

        // 调用 before_chat 钩子（带模型名称）
        let processed_message = if let Some(ref handler) = self.event_handler {
            match handler.before_chat_with_model(&message, model).await? {
                Some(msg) => msg,
                None => return Ok(String::new()),
            }
        } else {
            message
        };

        // 获取会话
        let session = self.get_session_arc(session_id).await?;

        // 发送消息
        let mut session_guard = session.write().await;
        let response = match session_guard.send(&processed_message).await {
            Ok(resp) => resp,
            Err(e) => {
                if let Some(ref handler) = self.event_handler
                    && let Some(fallback) = handler.on_error(&e).await?
                {
                    return Ok(fallback);
                }
                return Err(e);
            }
        };

        // 调用 after_chat 钩子（带元数据）
        let final_response = if let Some(ref handler) = self.event_handler {
            // 从会话中获取响应元数据
            let metadata = session_guard.last_response_metadata();
            if let Some(meta) = metadata {
                match handler.after_chat_with_metadata(&response, meta).await? {
                    Some(resp) => resp,
                    None => response,
                }
            } else {
                // 回退到旧方法（没有元数据）
                match handler.after_chat(&response).await? {
                    Some(resp) => resp,
                    None => response,
                }
            }
        } else {
            response
        };

        Ok(final_response)
    }

    /// Use the specified session and explicitly requested skill names for the prompt.
    pub async fn chat_with_session_and_skills(
        &self,
        session_id: &str,
        message: impl Into<String>,
        skill_names: &[String],
    ) -> LLMResult<String> {
        let message = message.into();

        let model = self.provider.default_model();
        let processed_message = if let Some(ref handler) = self.event_handler {
            match handler.before_chat_with_model(&message, model).await? {
                Some(msg) => msg,
                None => return Ok(String::new()),
            }
        } else {
            message
        };

        let session = self.get_session_arc(session_id).await?;
        let history = {
            let session_guard = session.read().await;
            session_guard.messages().to_vec()
        };

        let messages = self
            .build_context_messages(history, &processed_message, Some(skill_names))
            .await?;
        let response = self.send_messages(messages).await?;
        let metadata = super::types::LLMResponseMetadata::from(&response);
        let response_text = response
            .content()
            .ok_or_else(|| LLMError::Other("No content in response".to_string()))?
            .to_string();

        {
            let mut session_guard = session.write().await;
            Self::push_session_messages(
                &mut session_guard,
                &processed_message,
                &response_text,
                metadata.clone(),
            );
        }

        let final_response = if let Some(ref handler) = self.event_handler {
            match handler
                .after_chat_with_metadata(&response_text, &metadata)
                .await?
            {
                Some(resp) => resp,
                None => response_text,
            }
        } else {
            response_text
        };

        Ok(final_response)
    }

    /// 简单问答（不保留上下文）
    pub async fn ask(&self, question: impl Into<String>) -> LLMResult<String> {
        let question = question.into();

        if self.context_builder.is_some() {
            return self.ask_with_skills(question, &[]).await;
        }

        let mut builder = self.client.chat();

        // 使用动态 Prompt 模板（如果可用）
        let mut system_prompt = self.config.system_prompt.clone();

        if let Some(ref plugin) = self.prompt_plugin
            && let Some(template) = plugin.get_current_template().await
        {
            // 渲染默认模板（可以根据需要添加变量）
            match template.render(&[]) {
                Ok(prompt) => system_prompt = Some(prompt),
                Err(_) => {
                    // 如果渲染失败，使用回退的系统提示词
                    system_prompt = self.config.system_prompt.clone();
                }
            }
        }

        // 设置系统提示词
        if let Some(ref system) = system_prompt {
            builder = builder.system(system.clone());
        }

        if let Some(temp) = self.config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(tokens) = self.config.max_tokens {
            builder = builder.max_tokens(tokens);
        }

        builder = builder.user(question);

        // 添加工具
        if let Some(ref executor) = self.tool_executor {
            let tools = if self.tools.is_empty() {
                executor.available_tools().await?
            } else {
                self.tools.clone()
            };

            if !tools.is_empty() {
                builder = builder.tools(tools);
            }

            builder = builder.with_tool_executor(executor.clone());
            let response = builder.send_with_tools().await?;
            return response
                .content()
                .map(|s| s.to_string())
                .ok_or_else(|| LLMError::Other("No content in response".to_string()));
        }

        let response = builder.send().await?;
        response
            .content()
            .map(|s| s.to_string())
            .ok_or_else(|| LLMError::Other("No content in response".to_string()))
    }

    /// 简单问答（带显式 skills）
    pub async fn ask_with_skills(
        &self,
        question: impl Into<String>,
        skill_names: &[String],
    ) -> LLMResult<String> {
        let question = question.into();
        let messages = self
            .build_context_messages(Vec::new(), &question, Some(skill_names))
            .await?;
        let response = self.send_messages(messages).await?;
        response
            .content()
            .map(|s| s.to_string())
            .ok_or_else(|| LLMError::Other("No content in response".to_string()))
    }

    /// 设置 Prompt 场景
    pub async fn set_prompt_scenario(&self, scenario: impl Into<String>) {
        let scenario = scenario.into();

        if let Some(ref plugin) = self.prompt_plugin {
            plugin.set_active_scenario(&scenario).await;
        }
    }

    /// 清空对话历史（当前活动会话）
    pub async fn clear_history(&self) {
        let session_id = self.active_session_id.read().await.clone();
        let _ = self.clear_session_history(&session_id).await;
    }

    /// 清空指定会话的对话历史
    pub async fn clear_session_history(&self, session_id: &str) -> LLMResult<()> {
        let session = self.get_session_arc(session_id).await?;
        let mut session_guard = session.write().await;
        session_guard.clear();
        Ok(())
    }

    /// 获取对话历史（当前活动会话）
    pub async fn history(&self) -> Vec<ChatMessage> {
        let session_id = self.active_session_id.read().await.clone();
        self.get_session_history(&session_id)
            .await
            .unwrap_or_default()
    }

    /// 获取指定会话的对话历史
    pub async fn get_session_history(&self, session_id: &str) -> LLMResult<Vec<ChatMessage>> {
        let session = self.get_session_arc(session_id).await?;
        let session_guard = session.read().await;
        Ok(session_guard.messages().to_vec())
    }

    /// 设置工具
    pub fn set_tools(&mut self, tools: Vec<Tool>, executor: Arc<dyn ToolExecutor>) {
        self.tools = tools;
        self.tool_executor = Some(executor);

        // 更新 session 中的工具
        // 注意：这需要重新创建 session，因为 with_tools 消耗 self
    }

    /// 设置事件处理器
    pub fn set_event_handler(&mut self, handler: Box<dyn LLMAgentEventHandler>) {
        self.event_handler = Some(handler);
    }

    /// 向智能体添加插件
    pub fn add_plugin<P: AgentPlugin + 'static>(&mut self, plugin: P) {
        self.plugins.push(Box::new(plugin));
    }

    /// 向智能体添加插件列表
    pub fn add_plugins(&mut self, plugins: Vec<Box<dyn AgentPlugin>>) {
        self.plugins.extend(plugins);
    }

    // ========================================================================
    // 流式对话方法
    // ========================================================================

    /// 流式问答（不保留上下文）
    ///
    /// 返回一个 Stream，每次 yield 一个文本片段
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = agent.ask_stream("Tell me a story").await?;
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(text) => print!("{}", text),
    ///         Err(e) => einfo!("Error: {}", e),
    ///     }
    /// }
    /// ```
    pub async fn ask_stream(&self, question: impl Into<String>) -> LLMResult<TextStream> {
        let question = question.into();

        if self.context_builder.is_some() {
            return self.ask_stream_with_skills(question, &[]).await;
        }

        let mut builder = self.client.chat();

        if let Some(ref system) = self.config.system_prompt {
            builder = builder.system(system.clone());
        }

        if let Some(temp) = self.config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(tokens) = self.config.max_tokens {
            builder = builder.max_tokens(tokens);
        }

        builder = builder.user(question);

        // 发送流式请求
        let chunk_stream = builder.send_stream().await?;

        // 转换为纯文本流
        Ok(Self::chunk_stream_to_text_stream(chunk_stream))
    }

    /// 流式问答（带显式 skills）
    pub async fn ask_stream_with_skills(
        &self,
        question: impl Into<String>,
        skill_names: &[String],
    ) -> LLMResult<TextStream> {
        let question = question.into();
        let messages = self
            .build_context_messages(Vec::new(), &question, Some(skill_names))
            .await?;
        let chunk_stream = self.send_stream_messages(messages).await?;
        Ok(Self::chunk_stream_to_text_stream(chunk_stream))
    }

    /// 流式多轮对话（保留上下文）
    ///
    /// 注意：流式对话会在收到完整响应后更新历史记录
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = agent.chat_stream("Hello!").await?;
    /// let mut full_response = String::new();
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(text) => {
    ///             print!("{}", text);
    ///             full_response.push_str(&text);
    ///         }
    ///         Err(e) => einfo!("Error: {}", e),
    ///     }
    /// }
    /// info!();
    /// ```
    pub async fn chat_stream(&self, message: impl Into<String>) -> LLMResult<TextStream> {
        let session_id = self.active_session_id.read().await.clone();
        self.chat_stream_with_session(&session_id, message).await
    }

    /// 使用指定会话进行流式多轮对话
    ///
    /// # 参数
    /// - `session_id`: 会话唯一标识
    /// - `message`: 用户消息
    pub async fn chat_stream_with_session(
        &self,
        session_id: &str,
        message: impl Into<String>,
    ) -> LLMResult<TextStream> {
        let message = message.into();

        if self.context_builder.is_some() {
            return self
                .chat_stream_with_session_and_skills(session_id, message, &[])
                .await;
        }

        // 获取模型名称
        let model = self.provider.default_model();

        // 调用 before_chat 钩子（带模型名称）
        let processed_message = if let Some(ref handler) = self.event_handler {
            match handler.before_chat_with_model(&message, model).await? {
                Some(msg) => msg,
                None => return Ok(Box::pin(futures::stream::empty())),
            }
        } else {
            message
        };

        // 获取会话
        let session = self.get_session_arc(session_id).await?;

        // 获取当前历史
        let history = {
            let session_guard = session.read().await;
            session_guard.messages().to_vec()
        };

        // 构建请求
        let mut builder = self.client.chat();

        if let Some(ref system) = self.config.system_prompt {
            builder = builder.system(system.clone());
        }

        if let Some(temp) = self.config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(tokens) = self.config.max_tokens {
            builder = builder.max_tokens(tokens);
        }

        // 添加历史消息
        builder = builder.messages(history);
        builder = builder.user(processed_message.clone());

        // 发送流式请求
        let chunk_stream = builder.send_stream().await?;

        // 创建一个包装流，在完成时更新历史并调用事件处理
        let event_handler = self.event_handler.clone().map(Arc::new);
        let wrapped_stream = Self::create_history_updating_stream(
            chunk_stream,
            session,
            Some(processed_message),
            event_handler,
        );

        Ok(wrapped_stream)
    }

    /// 使用指定会话进行流式多轮对话（带显式 skills）
    pub async fn chat_stream_with_session_and_skills(
        &self,
        session_id: &str,
        message: impl Into<String>,
        skill_names: &[String],
    ) -> LLMResult<TextStream> {
        let message = message.into();

        let model = self.provider.default_model();
        let processed_message = if let Some(ref handler) = self.event_handler {
            match handler.before_chat_with_model(&message, model).await? {
                Some(msg) => msg,
                None => return Ok(Box::pin(futures::stream::empty())),
            }
        } else {
            message
        };

        let session = self.get_session_arc(session_id).await?;
        let history = {
            let session_guard = session.read().await;
            session_guard.messages().to_vec()
        };

        let messages = self
            .build_context_messages(history, &processed_message, Some(skill_names))
            .await?;
        let chunk_stream = self.send_stream_messages(messages).await?;

        let event_handler = self.event_handler.clone().map(Arc::new);
        Ok(Self::create_history_updating_stream(
            chunk_stream,
            session,
            Some(processed_message),
            event_handler,
        ))
    }

    /// 获取原始流式响应块（包含完整信息）
    ///
    /// 如果需要访问工具调用等详细信息，使用此方法
    pub async fn ask_stream_raw(&self, question: impl Into<String>) -> LLMResult<ChatStream> {
        let question = question.into();

        let mut builder = self.client.chat();

        if let Some(ref system) = self.config.system_prompt {
            builder = builder.system(system.clone());
        }

        if let Some(temp) = self.config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(tokens) = self.config.max_tokens {
            builder = builder.max_tokens(tokens);
        }

        builder = builder.user(question);

        builder.send_stream().await
    }

    /// 流式对话并收集完整响应（使用当前活动会话）
    ///
    /// 同时返回流和一个 channel 用于获取完整响应
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let (mut stream, full_response_rx) = agent.chat_stream_with_full("Hi").await?;
    ///
    /// while let Some(result) = stream.next().await {
    ///     if let Ok(text) = result {
    ///         print!("{}", text);
    ///     }
    /// }
    ///
    /// let full_response = full_response_rx.await?;
    /// info!("\nFull response: {}", full_response);
    /// ```
    pub async fn chat_stream_with_full(
        &self,
        message: impl Into<String>,
    ) -> LLMResult<(TextStream, tokio::sync::oneshot::Receiver<String>)> {
        let session_id = self.active_session_id.read().await.clone();
        self.chat_stream_with_full_session(&session_id, message)
            .await
    }

    /// 使用指定会话进行流式对话并收集完整响应
    ///
    /// # 参数
    /// - `session_id`: 会话唯一标识
    /// - `message`: 用户消息
    pub async fn chat_stream_with_full_session(
        &self,
        session_id: &str,
        message: impl Into<String>,
    ) -> LLMResult<(TextStream, tokio::sync::oneshot::Receiver<String>)> {
        let message = message.into();

        // 获取模型名称
        let model = self.provider.default_model();

        // 调用 before_chat 钩子（带模型名称）
        let processed_message = if let Some(ref handler) = self.event_handler {
            match handler.before_chat_with_model(&message, model).await? {
                Some(msg) => msg,
                None => {
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    let _ = tx.send(String::new());
                    return Ok((Box::pin(futures::stream::empty()), rx));
                }
            }
        } else {
            message
        };

        // 获取会话
        let session = self.get_session_arc(session_id).await?;

        // 获取当前历史
        let history = {
            let session_guard = session.read().await;
            session_guard.messages().to_vec()
        };

        let chunk_stream = if self.context_builder.is_some() {
            let requested_skills: &[String] = &[];
            let messages = self
                .build_context_messages(history, &processed_message, Some(requested_skills))
                .await?;
            self.send_stream_messages(messages).await?
        } else {
            let mut builder = self.client.chat();

            if let Some(ref system) = self.config.system_prompt {
                builder = builder.system(system.clone());
            }

            if let Some(temp) = self.config.temperature {
                builder = builder.temperature(temp);
            }

            if let Some(tokens) = self.config.max_tokens {
                builder = builder.max_tokens(tokens);
            }

            builder = builder.messages(history);
            builder = builder.user(processed_message.clone());
            builder.send_stream().await?
        };

        // 创建 channel 用于传递完整响应
        let (tx, rx) = tokio::sync::oneshot::channel();

        // 创建收集完整响应的流
        let event_handler = self.event_handler.clone().map(Arc::new);
        let wrapped_stream = Self::create_collecting_stream(
            chunk_stream,
            session,
            Some(processed_message),
            tx,
            event_handler,
        );

        Ok((wrapped_stream, rx))
    }

    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    /// 将 chunk stream 转换为纯文本 stream
    fn chunk_stream_to_text_stream(chunk_stream: ChatStream) -> TextStream {
        use futures::StreamExt;

        let text_stream = chunk_stream.filter_map(|result| async move {
            match result {
                Ok(chunk) => {
                    // 提取文本内容
                    if let Some(choice) = chunk.choices.first()
                        && let Some(ref content) = choice.delta.content
                        && !content.is_empty()
                    {
                        return Some(Ok(content.clone()));
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });

        Box::pin(text_stream)
    }

    /// 创建更新历史的流
    fn create_history_updating_stream(
        chunk_stream: ChatStream,
        session: Arc<RwLock<ChatSession>>,
        pending_user_message: Option<String>,
        event_handler: Option<Arc<Box<dyn LLMAgentEventHandler>>>,
    ) -> TextStream {
        use super::types::LLMResponseMetadata;

        let collected = Arc::new(tokio::sync::Mutex::new(String::new()));
        let collected_clone = collected.clone();
        let event_handler_clone = event_handler.clone();
        let metadata_collected = Arc::new(tokio::sync::Mutex::new(None::<LLMResponseMetadata>));
        let metadata_collected_clone = metadata_collected.clone();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        let stream = chunk_stream.filter_map(move |result| {
            let collected = collected.clone();
            let event_handler = event_handler.clone();
            let metadata_collected = metadata_collected.clone();
            let completed = completed.clone();
            async move {
                match result {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if choice.finish_reason.is_some() {
                                // 最后一个块包含 usage 数据，保存元数据
                                let metadata = LLMResponseMetadata::from(&chunk);
                                *metadata_collected.lock().await = Some(metadata);
                                completed.store(true, Ordering::Relaxed);
                                return None;
                            }
                            if let Some(ref content) = choice.delta.content
                                && !content.is_empty()
                            {
                                let mut collected = collected.lock().await;
                                collected.push_str(content);
                                return Some(Ok(content.clone()));
                            }
                        }
                        None
                    }
                    Err(e) => {
                        if let Some(handler) = event_handler {
                            let _ = handler.on_error(&e).await;
                        }
                        Some(Err(e))
                    }
                }
            }
        });

        let stream = stream
            .chain(futures::stream::once(async move {
                let full_response = collected_clone.lock().await.clone();
                let metadata = metadata_collected_clone.lock().await.clone();
                if completed_clone.load(Ordering::Relaxed) {
                    let mut session = session.write().await;
                    if let Some(user_message) = pending_user_message.as_ref() {
                        session.messages_mut().push(ChatMessage::user(user_message));
                    }
                    session
                        .messages_mut()
                        .push(ChatMessage::assistant(&full_response));

                    // 滑动窗口：裁剪历史消息以保持固定大小
                    let window_size = session.context_window_size();
                    if window_size.is_some() {
                        let current_messages = session.messages().to_vec();
                        *session.messages_mut() = ChatSession::apply_sliding_window_static(
                            &current_messages,
                            window_size,
                        );
                    }

                    if let Some(handler) = event_handler_clone {
                        if let Some(meta) = &metadata {
                            let _ = handler.after_chat_with_metadata(&full_response, meta).await;
                        } else {
                            let _ = handler.after_chat(&full_response).await;
                        }
                    }
                }
                Err(LLMError::Other("__stream_end__".to_string()))
            }))
            .filter_map(|result| async move {
                match result {
                    Ok(s) => Some(Ok(s)),
                    Err(LLMError::Other(message)) if message == "__stream_end__" => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Box::pin(stream)
    }

    /// 创建收集完整响应的流
    fn create_collecting_stream(
        chunk_stream: ChatStream,
        session: Arc<RwLock<ChatSession>>,
        pending_user_message: Option<String>,
        tx: tokio::sync::oneshot::Sender<String>,
        event_handler: Option<Arc<Box<dyn LLMAgentEventHandler>>>,
    ) -> TextStream {
        use super::types::LLMResponseMetadata;
        use futures::StreamExt;

        let collected = Arc::new(tokio::sync::Mutex::new(String::new()));
        let collected_clone = collected.clone();
        let event_handler_clone = event_handler.clone();
        let metadata_collected = Arc::new(tokio::sync::Mutex::new(None::<LLMResponseMetadata>));
        let metadata_collected_clone = metadata_collected.clone();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        let stream = chunk_stream.filter_map(move |result| {
            let collected = collected.clone();
            let event_handler = event_handler.clone();
            let metadata_collected = metadata_collected.clone();
            let completed = completed.clone();
            async move {
                match result {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if choice.finish_reason.is_some() {
                                // 最后一个块包含 usage 数据，保存元数据
                                let metadata = LLMResponseMetadata::from(&chunk);
                                *metadata_collected.lock().await = Some(metadata);
                                completed.store(true, Ordering::Relaxed);
                                return None;
                            }
                            if let Some(ref content) = choice.delta.content
                                && !content.is_empty()
                            {
                                let mut collected = collected.lock().await;
                                collected.push_str(content);
                                return Some(Ok(content.clone()));
                            }
                        }
                        None
                    }
                    Err(e) => {
                        if let Some(handler) = event_handler {
                            let _ = handler.on_error(&e).await;
                        }
                        Some(Err(e))
                    }
                }
            }
        });

        // 在流结束后更新历史并发送完整响应
        let stream = stream
            .chain(futures::stream::once(async move {
                let full_response = collected_clone.lock().await.clone();
                let mut processed_response = full_response.clone();
                let metadata = metadata_collected_clone.lock().await.clone();

                if completed_clone.load(Ordering::Relaxed) {
                    if let Some(handler) = event_handler_clone {
                        if let Some(meta) = &metadata {
                            if let Ok(Some(resp)) = handler
                                .after_chat_with_metadata(&processed_response, meta)
                                .await
                            {
                                processed_response = resp;
                            }
                        } else if let Ok(Some(resp)) = handler.after_chat(&processed_response).await
                        {
                            processed_response = resp;
                        }
                    }

                    let mut session = session.write().await;
                    if let Some(user_message) = pending_user_message.as_ref() {
                        session.messages_mut().push(ChatMessage::user(user_message));
                    }
                    session
                        .messages_mut()
                        .push(ChatMessage::assistant(&processed_response));

                    // 滑动窗口：裁剪历史消息以保持固定大小
                    let window_size = session.context_window_size();
                    if window_size.is_some() {
                        let current_messages = session.messages().to_vec();
                        *session.messages_mut() = ChatSession::apply_sliding_window_static(
                            &current_messages,
                            window_size,
                        );
                    }
                }

                let _ = tx.send(processed_response);

                Err(LLMError::Other("__stream_end__".to_string()))
            }))
            .filter_map(|result| async move {
                match result {
                    Ok(s) => Some(Ok(s)),
                    Err(LLMError::Other(message)) if message == "__stream_end__" => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Box::pin(stream)
    }
}

/// LLM Agent 构建器
pub struct LLMAgentBuilder {
    agent_id: String,
    name: Option<String>,
    provider: Option<Arc<dyn LLMProvider>>,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Vec<Tool>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    event_handler: Option<Box<dyn LLMAgentEventHandler>>,
    plugins: Vec<Box<dyn AgentPlugin>>,
    custom_config: HashMap<String, String>,
    prompt_plugin: Option<Box<dyn prompt::PromptTemplatePlugin>>,
    session_id: Option<String>,
    user_id: Option<String>,
    tenant_id: Option<String>,
    context_window_size: Option<usize>,
    /// 持久化存储（用于从数据库加载历史会话）
    message_store: Option<Arc<dyn crate::persistence::MessageStore + Send + Sync>>,
    session_store: Option<Arc<dyn crate::persistence::SessionStore + Send + Sync>>,
    persistence_user_id: Option<uuid::Uuid>,
    persistence_tenant_id: Option<uuid::Uuid>,
    persistence_agent_id: Option<uuid::Uuid>,
    workspace: Option<std::path::PathBuf>,
    skills_manager: Option<Arc<dyn PromptSkillsManager>>,
    always_skill_names: Vec<String>,
    context_builder: Option<AgentContextBuilder>,
}

impl Default for LLMAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMAgentBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            agent_id: uuid::Uuid::now_v7().to_string(),
            name: None,
            provider: None,
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            tools: Vec::new(),
            tool_executor: None,
            event_handler: None,
            plugins: Vec::new(),
            custom_config: HashMap::new(),
            prompt_plugin: None,
            session_id: None,
            user_id: None,
            tenant_id: None,
            context_window_size: None,
            message_store: None,
            session_store: None,
            persistence_user_id: None,
            persistence_tenant_id: None,
            persistence_agent_id: None,
            workspace: None,
            skills_manager: None,
            always_skill_names: Vec::new(),
            context_builder: None,
        }
    }

    /// 设置id
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = id.into();
        self
    }

    /// 设置名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 设置 LLM Provider
    pub fn with_provider(mut self, provider: Arc<dyn LLMProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// 设置系统提示词
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// 设置温度
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// 设置最大 token 数
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// 添加工具
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// 设置工具列表
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// 设置工具执行器
    pub fn with_tool_executor(mut self, executor: Arc<dyn ToolExecutor>) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    /// Set the workspace used by the prompt context builder.
    pub fn with_workspace(mut self, workspace: impl Into<std::path::PathBuf>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }

    /// Attach a skills manager for prompt-context based loading.
    pub fn with_skills_manager(mut self, skills: Arc<dyn PromptSkillsManager>) -> Self {
        self.skills_manager = Some(skills);
        self
    }

    /// Set skill names that should always be loaded into context.
    pub fn with_always_skills(mut self, names: Vec<String>) -> Self {
        self.always_skill_names = names;
        self
    }

    /// Attach a fully constructed context builder.
    pub fn with_context_builder(mut self, context_builder: AgentContextBuilder) -> Self {
        self.context_builder = Some(context_builder);
        self
    }

    /// 设置事件处理器
    pub fn with_event_handler(mut self, handler: Box<dyn LLMAgentEventHandler>) -> Self {
        self.event_handler = Some(handler);
        self
    }

    /// 添加插件
    pub fn with_plugin(mut self, plugin: impl AgentPlugin + 'static) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    /// 添加插件列表
    pub fn with_plugins(mut self, plugins: Vec<Box<dyn AgentPlugin>>) -> Self {
        self.plugins.extend(plugins);
        self
    }

    /// 添加持久化插件（便捷方法）
    ///
    /// 持久化插件实现了 AgentPlugin trait，同时也是一个 LLMAgentEventHandler，
    /// 会自动注册到 agent 的插件列表和事件处理器中。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use mofa_sdk::persistence::{PersistencePlugin, PostgresStore};
    /// use mofa_sdk::llm::LLMAgentBuilder;
    /// use std::sync::Arc;
    /// use uuid::Uuid;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let store = Arc::new(PostgresStore::connect("postgres://localhost/mofa").await?);
    /// let user_id = Uuid::now_v7();
    /// let tenant_id = Uuid::now_v7();
    /// let agent_id = Uuid::now_v7();
    /// let session_id = Uuid::now_v7();
    ///
    /// let plugin = PersistencePlugin::new(
    ///     "persistence-plugin",
    ///     store,
    ///     user_id,
    ///     tenant_id,
    ///     agent_id,
    ///     session_id,
    /// );
    ///
    /// let agent = LLMAgentBuilder::new()
    ///     .with_id("my-agent")
    ///     .with_persistence_plugin(plugin)
    ///     .build_async()
    ///     .await;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_persistence_plugin(
        mut self,
        plugin: crate::persistence::PersistencePlugin,
    ) -> Self {
        self.message_store = Some(plugin.message_store());
        self.session_store = plugin.session_store();
        self.persistence_user_id = Some(plugin.user_id());
        self.persistence_tenant_id = Some(plugin.tenant_id());
        self.persistence_agent_id = Some(plugin.agent_id());

        // 将持久化插件添加到插件列表
        // 同时作为事件处理器
        let plugin_box: Box<dyn AgentPlugin> = Box::new(plugin.clone());
        let event_handler: Box<dyn LLMAgentEventHandler> = Box::new(plugin);
        self.plugins.push(plugin_box);
        self.event_handler = Some(event_handler);
        self
    }

    /// 设置 Prompt 模板插件
    pub fn with_prompt_plugin(
        mut self,
        plugin: impl prompt::PromptTemplatePlugin + 'static,
    ) -> Self {
        self.prompt_plugin = Some(Box::new(plugin));
        self
    }

    /// 设置支持热重载的 Prompt 模板插件
    pub fn with_hot_reload_prompt_plugin(
        mut self,
        plugin: prompt::HotReloadableRhaiPromptPlugin,
    ) -> Self {
        self.prompt_plugin = Some(Box::new(plugin));
        self
    }

    /// 添加自定义配置
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_config.insert(key.into(), value.into());
        self
    }

    /// 设置初始会话 ID
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgentBuilder::new()
    ///     .with_id("my-agent")
    ///     .with_initial_session_id("user-session-001")
    ///     .build();
    /// ```
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// 设置用户 ID
    ///
    /// 用于数据库持久化和多用户场景的消息隔离。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgentBuilder::new()
    ///     .with_id("my-agent")
    ///     .with_user("user-123")
    ///     .build();
    /// ```
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// 设置租户 ID
    ///
    /// 用于多租户支持，实现不同租户的数据隔离。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgentBuilder::new()
    ///     .with_id("my-agent")
    ///     .with_tenant("tenant-abc")
    ///     .build();
    /// ```
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// 设置上下文窗口大小（滑动窗口）
    ///
    /// 用于滑动窗口消息管理，指定保留的最大对话轮数。
    /// 当消息历史超过此大小时，会自动裁剪较早的消息。
    ///
    /// # 参数
    /// - `size`: 上下文窗口大小（单位：轮数，rounds）
    ///
    /// # 注意
    /// - 单位是**轮数**（rounds），不是 token 数量
    /// - 每轮对话 ≈ 1 个用户消息 + 1 个助手响应
    /// - 系统消息始终保留，不计入轮数限制
    /// - 从数据库加载消息时也会应用此限制
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgentBuilder::new()
    ///     .with_id("my-agent")
    ///     .with_sliding_window(10)  // 只保留最近 10 轮对话
    ///     .build();
    /// ```
    pub fn with_sliding_window(mut self, size: usize) -> Self {
        self.context_window_size = Some(size);
        self
    }

    /// 从环境变量创建基础配置
    ///
    /// 自动配置：
    /// - OpenAI Provider（从 OPENAI_API_KEY）
    /// - 默认 temperature (0.7) 和 max_tokens (4096)
    ///
    /// # 环境变量
    /// - OPENAI_API_KEY: OpenAI API 密钥（必需）
    /// - OPENAI_BASE_URL: 可选的 API 基础 URL
    /// - OPENAI_MODEL: 可选的默认模型
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use mofa_sdk::llm::LLMAgentBuilder;
    ///
    /// let agent = LLMAgentBuilder::from_env()?
    ///     .with_system_prompt("You are a helpful assistant.")
    ///     .build();
    /// ```
    pub fn from_env() -> LLMResult<Self> {
        use super::openai::{OpenAIConfig, OpenAIProvider};

        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            LLMError::ConfigError("OPENAI_API_KEY environment variable not set".to_string())
        })?;

        let mut config = OpenAIConfig::new(api_key);

        if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            config = config.with_base_url(&base_url);
        }

        if let Ok(model) = std::env::var("OPENAI_MODEL") {
            config = config.with_model(&model);
        }

        Ok(Self::new()
            .with_provider(Arc::new(OpenAIProvider::with_config(config)))
            .with_temperature(0.7)
            .with_max_tokens(4096))
    }

    fn build_prompt_context_builder(&self) -> Option<AgentContextBuilder> {
        if let Some(context_builder) = &self.context_builder {
            return Some(context_builder.clone());
        }

        let skills = self.skills_manager.clone()?;
        let workspace = self
            .workspace
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));

        let identity = AgentIdentity::new(
            self.name.clone().unwrap_or_else(|| self.agent_id.clone()),
            "AI assistant",
        );

        Some(
            AgentContextBuilder::new(workspace)
                .with_identity(identity)
                .with_skills(skills)
                .with_always_skills(self.always_skill_names.clone()),
        )
    }

    /// 构建 LLM Agent
    ///
    /// # Panics
    /// 如果未设置 provider 则 panic
    pub fn build(self) -> LLMAgent {
        let context_builder = self.build_prompt_context_builder();
        let provider = self
            .provider
            .expect("LLM provider must be set before building");

        let config = LLMAgentConfig {
            agent_id: self.agent_id.clone(),
            name: self.name.unwrap_or_else(|| self.agent_id.clone()),
            system_prompt: self.system_prompt,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            custom_config: self.custom_config,
            user_id: self.user_id,
            tenant_id: self.tenant_id,
            context_window_size: self.context_window_size,
        };

        let mut agent = LLMAgent::with_initial_session(config, provider, self.session_id);

        // 设置Prompt模板插件
        agent.prompt_plugin = self.prompt_plugin;

        if let Some(executor) = self.tool_executor {
            agent.set_tools(self.tools, executor);
        }

        if let Some(handler) = self.event_handler {
            agent.set_event_handler(handler);
        }

        // 处理插件列表：提取 TTS 插件
        let mut plugins = self.plugins;
        let mut tts_plugin = None;

        // 查找并提取 TTS 插件
        for i in (0..plugins.len()).rev() {
            if plugins[i].as_any().is::<mofa_plugins::tts::TTSPlugin>() {
                // 使用 Any::downcast_ref 检查类型
                // 由于我们需要获取所有权，这里使用 is 检查后移除
                let plugin = plugins.remove(i);
                // 尝试 downcast
                if let Ok(tts) = plugin.into_any().downcast::<mofa_plugins::tts::TTSPlugin>() {
                    tts_plugin = Some(Arc::new(Mutex::new(*tts)));
                }
            }
        }

        // 添加剩余插件
        agent.add_plugins(plugins);

        // 设置 TTS 插件
        agent.tts_plugin = tts_plugin;
        agent.context_builder = context_builder;

        agent
    }

    /// 尝试构建 LLM Agent
    ///
    /// 如果未设置 provider 则返回错误
    pub fn try_build(self) -> LLMResult<LLMAgent> {
        let context_builder = self.build_prompt_context_builder();
        let provider = self
            .provider
            .ok_or_else(|| LLMError::ConfigError("LLM provider not set".to_string()))?;

        let config = LLMAgentConfig {
            agent_id: self.agent_id.clone(),
            name: self.name.unwrap_or_else(|| self.agent_id.clone()),
            system_prompt: self.system_prompt,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            custom_config: self.custom_config,
            user_id: self.user_id,
            tenant_id: self.tenant_id,
            context_window_size: self.context_window_size,
        };

        let mut agent = LLMAgent::with_initial_session(config, provider, self.session_id);

        if let Some(executor) = self.tool_executor {
            agent.set_tools(self.tools, executor);
        }

        if let Some(handler) = self.event_handler {
            agent.set_event_handler(handler);
        }

        // 处理插件列表：提取 TTS 插件
        let mut plugins = self.plugins;
        let mut tts_plugin = None;

        // 查找并提取 TTS 插件
        for i in (0..plugins.len()).rev() {
            if plugins[i].as_any().is::<mofa_plugins::tts::TTSPlugin>() {
                // 使用 Any::downcast_ref 检查类型
                // 由于我们需要获取所有权，这里使用 is 检查后移除
                let plugin = plugins.remove(i);
                // 尝试 downcast
                if let Ok(tts) = plugin.into_any().downcast::<mofa_plugins::tts::TTSPlugin>() {
                    tts_plugin = Some(Arc::new(Mutex::new(*tts)));
                }
            }
        }

        // 添加剩余插件
        agent.add_plugins(plugins);

        // 设置 TTS 插件
        agent.tts_plugin = tts_plugin;
        agent.context_builder = context_builder;

        Ok(agent)
    }

    /// 异步构建 LLM Agent（支持从数据库加载会话）
    ///
    /// 使用持久化插件加载会话历史。
    ///
    /// # 示例（使用持久化插件）
    ///
    /// ```rust,ignore
    /// use mofa_sdk::persistence::{PersistencePlugin, PostgresStore};
    ///
    /// let store = PostgresStore::connect("postgres://localhost/mofa").await?;
    /// let user_id = Uuid::now_v7();
    /// let tenant_id = Uuid::now_v7();
    /// let agent_id = Uuid::now_v7();
    /// let session_id = Uuid::now_v7();
    ///
    /// let plugin = PersistencePlugin::new(
    ///     "persistence-plugin",
    ///     Arc::new(store),
    ///     user_id,
    ///     tenant_id,
    ///     agent_id,
    ///     session_id,
    /// );
    ///
    /// let agent = LLMAgentBuilder::from_env()?
    ///     .with_system_prompt("You are helpful.")
    ///     .with_persistence_plugin(plugin)
    ///     .build_async()
    ///     .await;
    /// ```
    pub async fn build_async(mut self) -> LLMAgent {
        let context_builder = self.build_prompt_context_builder();
        let provider = self
            .provider
            .expect("LLM provider must be set before building");

        // Clone tenant_id for potential fallback use before moving into config
        let tenant_id_for_persistence = self.tenant_id.clone();

        let config = LLMAgentConfig {
            agent_id: self.agent_id.clone(),
            name: self.name.unwrap_or_else(|| self.agent_id.clone()),
            system_prompt: self.system_prompt,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            custom_config: self.custom_config,
            user_id: self.user_id,
            tenant_id: self.tenant_id,
            context_window_size: self.context_window_size,
        };

        // Fallback: If stores are set but persistence_tenant_id is None, use tenant_id
        let persistence_tenant_id = if self.session_store.is_some()
            && self.persistence_tenant_id.is_none()
            && let Some(ref tenant_id) = tenant_id_for_persistence
        {
            uuid::Uuid::parse_str(tenant_id).ok()
        } else {
            self.persistence_tenant_id
        };

        // 使用异步方法，支持从数据库加载
        let mut agent = LLMAgent::with_initial_session_async(
            config,
            provider,
            self.session_id,
            self.message_store,
            self.session_store,
            self.persistence_user_id,
            persistence_tenant_id,
            self.persistence_agent_id,
        )
        .await;

        // 设置Prompt模板插件
        agent.prompt_plugin = self.prompt_plugin;

        if self.tools.is_empty()
            && let Some(executor) = self.tool_executor.as_ref()
            && let Ok(tools) = executor.available_tools().await
        {
            self.tools = tools;
        }

        if let Some(executor) = self.tool_executor {
            agent.set_tools(self.tools, executor);
        }

        // 处理插件列表：
        // 1. 从持久化插件加载历史（新方式）
        // 2. 提取 TTS 插件
        let mut plugins = self.plugins;
        let mut tts_plugin = None;
        let history_loaded_from_plugin = false;

        // 查找并提取 TTS 插件
        for i in (0..plugins.len()).rev() {
            if plugins[i].as_any().is::<mofa_plugins::tts::TTSPlugin>() {
                // 使用 Any::downcast_ref 检查类型
                // 由于我们需要获取所有权，这里使用 is 检查后移除
                let plugin = plugins.remove(i);
                // 尝试 downcast
                if let Ok(tts) = plugin.into_any().downcast::<mofa_plugins::tts::TTSPlugin>() {
                    tts_plugin = Some(Arc::new(Mutex::new(*tts)));
                }
            }
        }

        // 从持久化插件加载历史（新方式）
        if !history_loaded_from_plugin {
            for plugin in &plugins {
                // 通过 metadata 识别持久化插件
                if plugin.metadata().plugin_type == PluginType::Storage
                    && plugin
                        .metadata()
                        .capabilities
                        .contains(&"message_persistence".to_string())
                {
                    // 这里我们无法直接调用泛型 PersistencePlugin 的 load_history
                    // 因为 trait object 无法访问泛型方法
                    // 历史加载将由 LLMAgent 在首次运行时通过 store 完成
                    tracing::info!("📦 检测到持久化插件，将在 agent 初始化后加载历史");
                    break;
                }
            }
        }

        // 添加剩余插件
        agent.add_plugins(plugins);

        // 设置 TTS 插件
        agent.tts_plugin = tts_plugin;

        // 设置事件处理器
        if let Some(handler) = self.event_handler {
            agent.set_event_handler(handler);
        }

        agent.context_builder = context_builder;

        agent
    }
}

// ============================================================================
// 从配置文件创建
// ============================================================================

impl LLMAgentBuilder {
    /// 从 agent.yml 配置文件创建 Builder
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use mofa_sdk::llm::LLMAgentBuilder;
    ///
    /// let agent = LLMAgentBuilder::from_config_file("agent.yml")?
    ///     .build();
    /// ```
    pub fn from_config_file(path: impl AsRef<std::path::Path>) -> LLMResult<Self> {
        let path = path.as_ref();
        let config = crate::config::AgentYamlConfig::from_file(path)
            .map_err(|e| LLMError::ConfigError(e.to_string()))?;
        let persisted_hub_config = crate::config::load_global_config()
            .map_err(|e| LLMError::ConfigError(e.to_string()))?;
        let workspace = path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        Self::from_yaml_config_with_sources(config, &workspace, Some(persisted_hub_config), true)
    }

    /// 从 YAML 配置创建 Builder
    pub fn from_yaml_config(config: crate::config::AgentYamlConfig) -> LLMResult<Self> {
        let workspace = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        Self::from_yaml_config_with_workspace(config, workspace)
    }

    /// 从 YAML 配置创建 Builder，并显式指定 skills 工作目录
    pub fn from_yaml_config_with_workspace(
        config: crate::config::AgentYamlConfig,
        workspace: impl AsRef<std::path::Path>,
    ) -> LLMResult<Self> {
        Self::from_yaml_config_with_sources(config, workspace, None, false)
    }

    fn from_yaml_config_with_sources(
        config: crate::config::AgentYamlConfig,
        workspace: impl AsRef<std::path::Path>,
        persisted_hub_config: Option<HashMap<String, String>>,
        include_env_hub_overrides: bool,
    ) -> LLMResult<Self> {
        let workspace = workspace.as_ref();
        let skills_config = config.skills.clone();
        let mut builder = Self::new()
            .with_id(&config.agent.id)
            .with_name(&config.agent.name);
        // 配置 LLM provider
        if let Some(llm_config) = config.llm {
            let provider = create_provider_from_config(&llm_config)?;
            builder = builder.with_provider(Arc::new(provider));

            if let Some(temp) = llm_config.temperature {
                builder = builder.with_temperature(temp);
            }
            if let Some(tokens) = llm_config.max_tokens {
                builder = builder.with_max_tokens(tokens);
            }
            if let Some(prompt) = llm_config.system_prompt {
                builder = builder.with_system_prompt(prompt);
            }
        }

        apply_skills_from_file_config(
            builder.with_workspace(workspace),
            workspace,
            skills_config.as_ref(),
            persisted_hub_config.as_ref(),
            include_env_hub_overrides,
        )
    }

    // ========================================================================
    // 数据库加载方法
    // ========================================================================

    /// 从数据库加载 agent 配置（全局查找）
    ///
    /// 根据 agent_code 从数据库加载 agent 配置及其关联的 provider。
    ///
    /// # 参数
    /// - `store`: 实现了 AgentStore 的持久化存储
    /// - `agent_code`: Agent 代码（唯一标识）
    ///
    /// # 错误
    /// - 如果 agent 不存在
    /// - 如果 agent 被禁用 (agent_status = false)
    /// - 如果 provider 被禁用 (enabled = false)
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use mofa_sdk::{llm::LLMAgentBuilder, persistence::PostgresStore};
    ///
    /// let store = PostgresStore::from_env().await?;
    /// let agent = LLMAgentBuilder::from_database(&store, "my-agent").await?.build();
    /// ```
    #[cfg(feature = "persistence-postgres")]
    pub async fn from_database<S>(store: &S, agent_code: &str) -> LLMResult<Self>
    where
        S: crate::persistence::AgentStore + Send + Sync,
    {
        let config = store
            .get_agent_by_code_with_provider(agent_code)
            .await
            .map_err(|e| LLMError::Other(format!("Failed to load agent from database: {}", e)))?
            .ok_or_else(|| {
                LLMError::Other(format!(
                    "Agent with code '{}' not found in database",
                    agent_code
                ))
            })?;

        Self::from_agent_config(&config)
    }

    /// 从数据库加载 agent 配置（租户隔离）
    ///
    /// 根据 tenant_id 和 agent_code 从数据库加载 agent 配置及其关联的 provider。
    ///
    /// # 参数
    /// - `store`: 实现了 AgentStore 的持久化存储
    /// - `tenant_id`: 租户 ID
    /// - `agent_code`: Agent 代码
    ///
    /// # 错误
    /// - 如果 agent 不存在
    /// - 如果 agent 被禁用 (agent_status = false)
    /// - 如果 provider 被禁用 (enabled = false)
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use mofa_sdk::{llm::LLMAgentBuilder, persistence::PostgresStore};
    /// use uuid::Uuid;
    ///
    /// let store = PostgresStore::from_env().await?;
    /// let tenant_id = Uuid::parse_str("xxx-xxx-xxx")?;
    /// let agent = LLMAgentBuilder::from_database_with_tenant(&store, tenant_id, "my-agent").await?.build();
    /// ```
    #[cfg(feature = "persistence-postgres")]
    pub async fn from_database_with_tenant<S>(
        store: &S,
        tenant_id: uuid::Uuid,
        agent_code: &str,
    ) -> LLMResult<Self>
    where
        S: crate::persistence::AgentStore + Send + Sync,
    {
        let config = store
            .get_agent_by_code_and_tenant_with_provider(tenant_id, agent_code)
            .await
            .map_err(|e| LLMError::Other(format!("Failed to load agent from database: {}", e)))?
            .ok_or_else(|| {
                LLMError::Other(format!(
                    "Agent with code '{}' not found for tenant {}",
                    agent_code, tenant_id
                ))
            })?;

        Self::from_agent_config(&config)
    }

    /// 使用数据库 agent 配置，但允许进一步定制
    ///
    /// 加载数据库配置后，可以继续使用 builder 方法进行定制。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let agent = LLMAgentBuilder::with_database_agent(&store, "my-agent")
    ///     .await?
    ///     .with_temperature(0.8)  // 覆盖数据库中的温度设置
    ///     .with_system_prompt("Custom prompt")  // 覆盖系统提示词
    ///     .build();
    /// ```
    #[cfg(feature = "persistence-postgres")]
    pub async fn with_database_agent<S>(store: &S, agent_code: &str) -> LLMResult<Self>
    where
        S: crate::persistence::AgentStore + Send + Sync,
    {
        Self::from_database(store, agent_code).await
    }

    /// 从 AgentConfig 创建 Builder（内部辅助方法）
    #[cfg(feature = "persistence-postgres")]
    pub fn from_agent_config(config: &crate::persistence::AgentConfig) -> LLMResult<Self> {
        use super::openai::{OpenAIConfig, OpenAIProvider};

        let agent = &config.agent;
        let provider = &config.provider;

        // 检查 agent 是否启用
        if !agent.agent_status {
            return Err(LLMError::Other(format!(
                "Agent '{}' is disabled (agent_status = false)",
                agent.agent_code
            )));
        }

        // 检查 provider 是否启用
        if !provider.enabled {
            return Err(LLMError::Other(format!(
                "Provider '{}' is disabled (enabled = false)",
                provider.provider_name
            )));
        }

        // 根据 provider_type 创建 LLM Provider
        let llm_provider: Arc<dyn super::LLMProvider> = match provider.provider_type.as_str() {
            "openai" | "azure" | "compatible" | "local" => {
                let mut openai_config = OpenAIConfig::new(provider.api_key.clone());
                openai_config = openai_config.with_base_url(&provider.api_base);
                openai_config = openai_config.with_model(&agent.model_name);

                if let Some(temp) = agent.temperature {
                    openai_config = openai_config.with_temperature(temp);
                }

                if let Some(max_tokens) = agent.max_completion_tokens {
                    openai_config = openai_config.with_max_tokens(max_tokens as u32);
                }

                Arc::new(OpenAIProvider::with_config(openai_config))
            }
            "anthropic" => {
                let mut cfg = AnthropicConfig::new(provider.api_key.clone());
                cfg = cfg.with_base_url(&provider.api_base);
                cfg = cfg.with_model(&agent.model_name);

                if let Some(temp) = agent.temperature {
                    cfg = cfg.with_temperature(temp);
                }
                if let Some(tokens) = agent.max_completion_tokens {
                    cfg = cfg.with_max_tokens(tokens as u32);
                }

                Arc::new(AnthropicProvider::with_config(cfg))
            }
            "gemini" => {
                let mut cfg = GeminiConfig::new(provider.api_key.clone());
                cfg = cfg.with_base_url(&provider.api_base);
                cfg = cfg.with_model(&agent.model_name);

                if let Some(temp) = agent.temperature {
                    cfg = cfg.with_temperature(temp);
                }
                if let Some(tokens) = agent.max_completion_tokens {
                    cfg = cfg.with_max_tokens(tokens as u32);
                }

                Arc::new(GeminiProvider::with_config(cfg))
            }
            "ollama" => {
                let mut ollama_config = OllamaConfig::new();
                ollama_config = ollama_config.with_base_url(&provider.api_base);
                ollama_config = ollama_config.with_model(&agent.model_name);

                if let Some(temp) = agent.temperature {
                    ollama_config = ollama_config.with_temperature(temp);
                }

                if let Some(max_tokens) = agent.max_completion_tokens {
                    ollama_config = ollama_config.with_max_tokens(max_tokens as u32);
                }

                Arc::new(OllamaProvider::with_config(ollama_config))
            }
            other => {
                return Err(LLMError::Other(format!(
                    "Unsupported provider type: {}",
                    other
                )));
            }
        };

        // 创建基础 builder
        let mut builder = Self::new()
            .with_id(agent.id)
            .with_name(agent.agent_name.clone())
            .with_provider(llm_provider)
            .with_system_prompt(agent.system_prompt.clone())
            .with_tenant(agent.tenant_id.to_string());

        // 设置可选参数
        if let Some(temp) = agent.temperature {
            builder = builder.with_temperature(temp);
        }
        if let Some(tokens) = agent.max_completion_tokens {
            builder = builder.with_max_tokens(tokens as u32);
        }
        if let Some(limit) = agent.context_limit {
            builder = builder.with_sliding_window(limit as usize);
        }

        // 处理 custom_params (JSONB) - 将每个 key-value 添加到 custom_config
        if let Some(ref params) = agent.custom_params
            && let Some(obj) = params.as_object()
        {
            for (key, value) in obj.iter() {
                let value_str: String = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => value.to_string(),
                };
                builder = builder.with_config(key.as_str(), value_str);
            }
        }

        // 处理 response_format
        if let Some(ref format) = agent.response_format {
            builder = builder.with_config("response_format", format);
        }

        // 处理 stream
        if let Some(stream) = agent.stream {
            builder = builder.with_config("stream", if stream { "true" } else { "false" });
        }

        Ok(builder)
    }
}

/// 从配置创建 LLM Provider
fn create_provider_from_config(
    config: &crate::config::LLMYamlConfig,
) -> LLMResult<super::openai::OpenAIProvider> {
    use super::openai::{OpenAIConfig, OpenAIProvider};

    match config.provider.as_str() {
        "openai" => {
            let api_key = config
                .api_key
                .clone()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| LLMError::ConfigError("OpenAI API key not set".to_string()))?;

            let mut openai_config = OpenAIConfig::new(api_key);

            if let Some(ref model) = config.model {
                openai_config = openai_config.with_model(model);
            }
            if let Some(ref base_url) = config.base_url {
                openai_config = openai_config.with_base_url(base_url);
            }
            if let Some(temp) = config.temperature {
                openai_config = openai_config.with_temperature(temp);
            }
            if let Some(tokens) = config.max_tokens {
                openai_config = openai_config.with_max_tokens(tokens);
            }

            Ok(OpenAIProvider::with_config(openai_config))
        }
        "azure" => {
            let endpoint = config.base_url.clone().ok_or_else(|| {
                LLMError::ConfigError("Azure endpoint (base_url) not set".to_string())
            })?;
            let api_key = config
                .api_key
                .clone()
                .or_else(|| std::env::var("AZURE_OPENAI_API_KEY").ok())
                .ok_or_else(|| LLMError::ConfigError("Azure API key not set".to_string()))?;
            let deployment = config
                .deployment
                .clone()
                .or_else(|| config.model.clone())
                .ok_or_else(|| {
                    LLMError::ConfigError("Azure deployment name not set".to_string())
                })?;

            Ok(OpenAIProvider::azure(endpoint, api_key, deployment))
        }
        "compatible" | "local" => {
            let base_url = config.base_url.clone().ok_or_else(|| {
                LLMError::ConfigError("base_url not set for compatible provider".to_string())
            })?;
            let model = config
                .model
                .clone()
                .unwrap_or_else(|| "default".to_string());

            Ok(OpenAIProvider::local(base_url, model))
        }
        other => Err(LLMError::ConfigError(format!(
            "Unknown provider: {}",
            other
        ))),
    }
}

// ============================================================================
// MoFAAgent 实现 - 新的统一微内核架构
// ============================================================================

#[async_trait::async_trait]
impl mofa_kernel::agent::MoFAAgent for LLMAgent {
    fn id(&self) -> &str {
        &self.metadata.id
    }

    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn capabilities(&self) -> &mofa_kernel::agent::AgentCapabilities {
        // 将 metadata 中的 capabilities 转换为 AgentCapabilities
        // 这里需要使用一个静态的 AgentCapabilities 实例
        // 或者在 LLMAgent 中存储一个 AgentCapabilities 字段
        // 为了简化，我们创建一个基于当前 metadata 的实现
        use mofa_kernel::agent::AgentCapabilities;

        // 注意：这里返回的是一个临时引用，实际使用中可能需要调整 LLMAgent 的结构
        // 来存储一个 AgentCapabilities 实例
        // 这里我们使用一个 hack 来返回一个静态实例
        static CAPABILITIES: std::sync::OnceLock<AgentCapabilities> = std::sync::OnceLock::new();

        CAPABILITIES.get_or_init(|| {
            AgentCapabilities::builder()
                .tag("llm")
                .tag("chat")
                .tag("text-generation")
                .input_type(mofa_kernel::agent::InputType::Text)
                .output_type(mofa_kernel::agent::OutputType::Text)
                .supports_streaming(true)
                .supports_tools(true)
                .build()
        })
    }

    async fn initialize(
        &mut self,
        ctx: &mofa_kernel::agent::AgentContext,
    ) -> mofa_kernel::agent::AgentResult<()> {
        // 初始化所有插件（load -> init）
        let mut plugin_config = mofa_kernel::plugin::PluginConfig::new();
        for (k, v) in &self.config.custom_config {
            plugin_config.set(k, v);
        }
        if let Some(user_id) = &self.config.user_id {
            plugin_config.set("user_id", user_id);
        }
        if let Some(tenant_id) = &self.config.tenant_id {
            plugin_config.set("tenant_id", tenant_id);
        }
        let session_id = self.active_session_id.read().await.clone();
        plugin_config.set("session_id", session_id);

        let plugin_ctx =
            mofa_kernel::plugin::PluginContext::new(self.id()).with_config(plugin_config);

        for plugin in &mut self.plugins {
            plugin
                .load(&plugin_ctx)
                .await
                .map_err(|e| mofa_kernel::agent::AgentError::InitializationFailed(e.to_string()))?;
            plugin
                .init_plugin()
                .await
                .map_err(|e| mofa_kernel::agent::AgentError::InitializationFailed(e.to_string()))?;
        }
        self.state = mofa_kernel::agent::AgentState::Ready;

        // 将上下文信息保存到 metadata（如果需要）
        let _ = ctx;

        Ok(())
    }

    async fn execute(
        &mut self,
        input: mofa_kernel::agent::AgentInput,
        _ctx: &mofa_kernel::agent::AgentContext,
    ) -> mofa_kernel::agent::AgentResult<mofa_kernel::agent::AgentOutput> {
        use mofa_kernel::agent::{AgentError, AgentInput, AgentOutput};

        // 将 AgentInput 转换为字符串
        let message = match input {
            AgentInput::Text(text) => text,
            AgentInput::Json(json) => json.to_string(),
            _ => {
                return Err(AgentError::ValidationFailed(
                    "Unsupported input type for LLMAgent".to_string(),
                ));
            }
        };

        // 执行 chat
        let response = self
            .chat(&message)
            .await
            .map_err(|e| AgentError::ExecutionFailed(format!("LLM chat failed: {}", e)))?;

        // 将响应转换为 AgentOutput
        Ok(AgentOutput::text(response))
    }

    async fn shutdown(&mut self) -> mofa_kernel::agent::AgentResult<()> {
        // 销毁所有插件
        for plugin in &mut self.plugins {
            plugin
                .unload()
                .await
                .map_err(|e| mofa_kernel::agent::AgentError::ShutdownFailed(e.to_string()))?;
        }
        self.state = mofa_kernel::agent::AgentState::Shutdown;
        Ok(())
    }

    fn state(&self) -> mofa_kernel::agent::AgentState {
        self.state.clone()
    }
}

// ============================================================================
// 便捷函数
// ============================================================================

/// 快速创建简单的 LLM Agent
///
/// # 示例
///
/// ```rust,ignore
/// use mofa_sdk::llm::{simple_llm_agent, openai_from_env};
/// use std::sync::Arc;
///
/// let agent = simple_llm_agent(
///     "my-agent",
///     Arc::new(openai_from_env()),
///     "You are a helpful assistant."
/// );
/// ```
pub fn simple_llm_agent(
    agent_id: impl Into<String>,
    provider: Arc<dyn LLMProvider>,
    system_prompt: impl Into<String>,
) -> LLMAgent {
    LLMAgentBuilder::new()
        .with_id(agent_id)
        .with_provider(provider)
        .with_system_prompt(system_prompt)
        .build()
}

/// 从配置文件创建 LLM Agent
///
/// # 示例
///
/// ```rust,ignore
/// use mofa_sdk::llm::agent_from_config;
///
/// let agent = agent_from_config("agent.yml")?;
/// ```
pub fn agent_from_config(path: impl AsRef<std::path::Path>) -> LLMResult<LLMAgent> {
    LLMAgentBuilder::from_config_file(path)?.try_build()
}

fn apply_skills_from_file_config(
    builder: LLMAgentBuilder,
    workspace: &std::path::Path,
    skills_config: Option<&crate::config::SkillsYamlConfig>,
    persisted_hub_config: Option<&HashMap<String, String>>,
    include_env_hub_overrides: bool,
) -> LLMResult<LLMAgentBuilder> {
    let Some(skills_config) = skills_config else {
        return Ok(builder);
    };

    let mut search_dirs: Vec<std::path::PathBuf> = if skills_config.search_dirs.is_empty() {
        vec![workspace.join("skills")]
    } else {
        skills_config
            .search_dirs
            .iter()
            .map(std::path::PathBuf::from)
            .map(|dir| {
                if dir.is_absolute() {
                    dir
                } else {
                    workspace.join(dir)
                }
            })
            .collect()
    };

    if let Some(builtin_dir) = crate::skills::SkillsManager::find_builtin_skills()
        && !search_dirs.contains(&builtin_dir)
    {
        search_dirs.push(builtin_dir);
    }

    let manager = if let Some(hub) = &skills_config.hub {
        let hub_config =
            resolve_skill_hub_config(hub, persisted_hub_config, include_env_hub_overrides)?;

        crate::skills::SkillsManager::with_search_dirs_and_hub(search_dirs, hub_config)
            .map_err(|e| LLMError::ConfigError(e.to_string()))?
    } else if search_dirs.len() == 1 {
        crate::skills::SkillsManager::new(&search_dirs[0])
            .map_err(|e| LLMError::ConfigError(e.to_string()))?
    } else {
        crate::skills::SkillsManager::with_search_dirs(search_dirs)
            .map_err(|e| LLMError::ConfigError(e.to_string()))?
    };

    Ok(builder
        .with_skills_manager(Arc::new(manager))
        .with_always_skills(skills_config.always_load.clone()))
}

fn resolve_skill_hub_config(
    hub: &crate::config::SkillHubYamlConfig,
    persisted_hub_config: Option<&HashMap<String, String>>,
    include_env_hub_overrides: bool,
) -> LLMResult<crate::skills::SkillHubClientConfig> {
    let mut config =
        crate::skills::SkillHubClientConfig::new(crate::skills::DEFAULT_SKILLS_HUB_CATALOG_URL);

    if let Some(catalog_url) = persisted_hub_config.and_then(|persisted| {
        persisted
            .get(HUB_CONFIG_KEY_CATALOG_URL)
            .filter(|value| !value.trim().is_empty())
    }) {
        config.catalog_url = catalog_url.clone();
    }

    if let Some(value) = persisted_hub_config
        .and_then(|persisted| persisted.get(HUB_CONFIG_KEY_AUTO_INSTALL))
        .and_then(|value| parse_global_bool(value))
    {
        config.auto_install_on_miss = value;
    }

    if let Some(value) = persisted_hub_config
        .and_then(|persisted| persisted.get(HUB_CONFIG_KEY_COMPATIBILITY_TARGETS))
    {
        let targets = crate::config::parse_global_string_list(value);
        if !targets.is_empty() {
            config.compatibility_targets = targets;
        }
    }

    if include_env_hub_overrides {
        if let Ok(catalog_url) = std::env::var("MOFA_SKILLS_HUB_CATALOG_URL")
            && !catalog_url.trim().is_empty()
        {
            config.catalog_url = catalog_url;
        }
        if let Ok(auto_install) = std::env::var("MOFA_SKILLS_HUB_AUTO_INSTALL")
            && let Some(enabled) = parse_global_bool(&auto_install)
        {
            config.auto_install_on_miss = enabled;
        }
    }

    if let Some(catalog_url) = &hub.catalog_url {
        config.catalog_url = catalog_url.clone();
    }
    config.auto_install_on_miss = hub.auto_install;
    if !hub.compatibility_targets.is_empty() {
        config.compatibility_targets = hub.compatibility_targets.clone();
    }

    Ok(config)
}

fn parse_global_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
