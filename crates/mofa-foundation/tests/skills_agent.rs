use async_trait::async_trait;
use futures::{StreamExt, stream};
use mofa_foundation::llm::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice,
    ChunkChoice, ChunkDelta, FinishReason, LLMAgentBuilder, LLMError, LLMProvider, LLMResult,
    MessageContent, Role, SkillsManager, Usage,
};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Mutex;

#[derive(Default)]
struct StaticSkillsManager;

#[async_trait]
impl SkillsManager for StaticSkillsManager {
    async fn get_always_skills(&self) -> Vec<String> {
        vec!["catalog".to_string()]
    }

    async fn load_skills_for_context(&self, names: &[String]) -> String {
        names
            .iter()
            .map(|name| format!("Skill<{name}>"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn build_skills_summary(&self) -> String {
        "<skills><skill><name>catalog</name></skill></skills>".to_string()
    }
}

struct RecordingProvider {
    last_request: Arc<Mutex<Option<ChatCompletionRequest>>>,
}

impl RecordingProvider {
    fn new() -> Self {
        Self {
            last_request: Arc::new(Mutex::new(None)),
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

struct ErroringStreamProvider;

#[async_trait]
impl LLMProvider for RecordingProvider {
    fn name(&self) -> &str {
        "recording"
    }

    fn default_model(&self) -> &str {
        "recording-model"
    }

    fn supports_streaming(&self) -> bool {
        true
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

    async fn chat_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> LLMResult<mofa_foundation::llm::ChatStream> {
        *self.last_request.lock().await = Some(request);

        Ok(Box::pin(stream::iter(vec![
            Ok(ChatCompletionChunk {
                id: "chunk-1".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 0,
                model: "recording-model".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some(Role::Assistant),
                        content: Some("ok".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            }),
            Ok(ChatCompletionChunk {
                id: "chunk-2".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 0,
                model: "recording-model".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                }],
                usage: None,
            }),
        ])))
    }
}

#[async_trait]
impl LLMProvider for ErroringStreamProvider {
    fn name(&self) -> &str {
        "erroring"
    }

    fn default_model(&self) -> &str {
        "erroring-model"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat(&self, _request: ChatCompletionRequest) -> LLMResult<ChatCompletionResponse> {
        Err(LLMError::Other("chat should not be called".to_string()))
    }

    async fn chat_stream(
        &self,
        _request: ChatCompletionRequest,
    ) -> LLMResult<mofa_foundation::llm::ChatStream> {
        Ok(Box::pin(stream::iter(vec![Err(LLMError::Other(
            "stream failed".to_string(),
        ))])))
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

fn system_message_count(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .filter(|message| message.role == mofa_foundation::llm::Role::System)
        .count()
}

#[tokio::test]
async fn test_skill_aware_ask_includes_always_loaded_skills() {
    let workspace = TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::new());

    let agent = LLMAgentBuilder::new()
        .with_provider(provider.clone())
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let response = agent.ask("hello").await.unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("Base prompt"));
    assert!(system_prompt.contains("Skill<catalog>"));
}

#[tokio::test]
async fn test_ask_with_skills_includes_requested_skill_content() {
    let workspace = TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::new());

    let agent = LLMAgentBuilder::new()
        .with_provider(provider.clone())
        .with_workspace(workspace.path())
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let requested = vec!["remote".to_string()];
    let response = agent.ask_with_skills("hello", &requested).await.unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("Skill<catalog>"));
    assert!(system_prompt.contains("Skill<remote>"));
}

#[tokio::test]
async fn test_skill_aware_session_chat_sends_one_system_message_per_turn() {
    let workspace = TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::new());

    let agent = LLMAgentBuilder::new()
        .with_provider(provider.clone())
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let session_id = agent.create_session().await;
    let response = agent.chat_with_session(&session_id, "hello").await.unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    assert_eq!(system_message_count(&request), 1);
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("Base prompt"));
    assert!(system_prompt.contains("Skill<catalog>"));
}

#[tokio::test]
async fn test_chat_with_session_and_skills_sends_one_system_message_per_turn() {
    let workspace = TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::new());

    let agent = LLMAgentBuilder::new()
        .with_provider(provider.clone())
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let session_id = agent.create_session().await;
    let requested = vec!["remote".to_string()];
    let response = agent
        .chat_with_session_and_skills(&session_id, "hello", &requested)
        .await
        .unwrap();
    assert_eq!(response, "ok");

    let request = provider.take_request().await;
    assert_eq!(system_message_count(&request), 1);
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("Base prompt"));
    assert!(system_prompt.contains("Skill<catalog>"));
    assert!(system_prompt.contains("Skill<remote>"));
}

#[tokio::test]
async fn test_chat_stream_with_session_and_skills_sends_one_system_message_per_turn() {
    let workspace = TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::new());

    let agent = LLMAgentBuilder::new()
        .with_provider(provider.clone())
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let session_id = agent.create_session().await;
    let requested = vec!["remote".to_string()];
    let stream = agent
        .chat_stream_with_session_and_skills(&session_id, "hello", &requested)
        .await
        .unwrap();
    let chunks = stream.collect::<Vec<_>>().await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].as_ref().unwrap(), "ok");

    let request = provider.take_request().await;
    assert_eq!(system_message_count(&request), 1);
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("Base prompt"));
    assert!(system_prompt.contains("Skill<catalog>"));
    assert!(system_prompt.contains("Skill<remote>"));
}

#[tokio::test]
async fn test_chat_stream_with_full_session_is_skill_aware() {
    let workspace = TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::new());

    let agent = LLMAgentBuilder::new()
        .with_provider(provider.clone())
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let session_id = agent.create_session().await;
    let (stream, full_response_rx) = agent
        .chat_stream_with_full_session(&session_id, "hello")
        .await
        .unwrap();
    let chunks = stream.collect::<Vec<_>>().await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].as_ref().unwrap(), "ok");
    assert_eq!(full_response_rx.await.unwrap(), "ok");

    let request = provider.take_request().await;
    assert_eq!(system_message_count(&request), 1);
    let system_prompt = system_text(&request);
    assert!(system_prompt.contains("Base prompt"));
    assert!(system_prompt.contains("Skill<catalog>"));
}

#[tokio::test]
async fn test_chat_stream_with_session_and_skills_does_not_mutate_history_on_error() {
    let workspace = TempDir::new().unwrap();

    let agent = LLMAgentBuilder::new()
        .with_provider(Arc::new(ErroringStreamProvider))
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let session_id = agent.create_session().await;
    let before = agent.get_session_history(&session_id).await.unwrap();

    let requested = vec!["remote".to_string()];
    let stream = agent
        .chat_stream_with_session_and_skills(&session_id, "hello", &requested)
        .await
        .unwrap();
    let chunks = stream.collect::<Vec<_>>().await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].as_ref().unwrap_err().to_string(),
        "LLM error: stream failed"
    );

    let after = agent.get_session_history(&session_id).await.unwrap();
    assert_eq!(after.len(), before.len());
    assert!(after.iter().all(|message| message.role != Role::User));
}

#[tokio::test]
async fn test_chat_stream_with_full_session_does_not_mutate_history_on_error() {
    let workspace = TempDir::new().unwrap();

    let agent = LLMAgentBuilder::new()
        .with_provider(Arc::new(ErroringStreamProvider))
        .with_workspace(workspace.path())
        .with_system_prompt("Base prompt")
        .with_skills_manager(Arc::new(StaticSkillsManager))
        .try_build()
        .unwrap();

    let session_id = agent.create_session().await;
    let before = agent.get_session_history(&session_id).await.unwrap();

    let (stream, full_response_rx) = agent
        .chat_stream_with_full_session(&session_id, "hello")
        .await
        .unwrap();
    let chunks = stream.collect::<Vec<_>>().await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].as_ref().unwrap_err().to_string(),
        "LLM error: stream failed"
    );
    assert_eq!(full_response_rx.await.unwrap(), "");

    let after = agent.get_session_history(&session_id).await.unwrap();
    assert_eq!(after.len(), before.len());
    assert!(after.iter().all(|message| message.role != Role::User));
}
