use std::{
    collections::HashSet,
    error::Error,
    sync::{Arc, Mutex as StdMutex},
};

use async_trait::async_trait;
#[cfg(feature = "mcp-demo")]
use http::{HeaderName, HeaderValue};
use langchain_rust::{
    schemas::{
        memory::{BaseMemory, MemoryError},
        Message, MessageType,
    },
    tools::Tool,
};
#[cfg(feature = "mcp-demo")]
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{
    model::{CallToolRequestParams, ClientCapabilities, ClientConfig, Implementation},
    service::{ClientLifecycleMode, ClientServiceExt, Peer, RoleClient, RunningService},
    transport::{
        child_process::TokioChildProcess, streamable_http_client::StreamableHttpClientTransport,
    },
};
use rusqlite::{params, Connection};
use serde_json::Value;
use sha2::{Digest, Sha256};
#[cfg(feature = "mcp-demo")]
use std::collections::HashMap;
use tokio::process::Command;

const MEMORY_TABLE: &str = "langchain_agent_messages";

pub fn parse_tool_input(input: &str) -> Value {
    serde_json::from_str(input).unwrap_or_else(|_| Value::String(input.to_owned()))
}

struct ToolCallState {
    max_calls: usize,
    calls: usize,
    seen: HashSet<(String, Vec<u8>)>,
}

pub struct AgentToolCallGuard {
    state: Arc<StdMutex<ToolCallState>>,
}

impl AgentToolCallGuard {
    pub fn new(max_calls: usize) -> Self {
        Self {
            state: Arc::new(StdMutex::new(ToolCallState {
                max_calls,
                calls: 0,
                seen: HashSet::new(),
            })),
        }
    }

    pub fn wrap(&self, tool: Arc<dyn Tool>) -> Arc<dyn Tool> {
        Arc::new(GuardedTool {
            tool,
            state: self.state.clone(),
        })
    }
}

struct GuardedTool {
    tool: Arc<dyn Tool>,
    state: Arc<StdMutex<ToolCallState>>,
}

#[async_trait]
impl Tool for GuardedTool {
    fn name(&self) -> String {
        self.tool.name()
    }

    fn description(&self) -> String {
        self.tool.description()
    }

    fn parameters(&self) -> Value {
        self.tool.parameters()
    }

    async fn call(&self, input: &str) -> Result<String, Box<dyn Error>> {
        self.run(parse_tool_input(input)).await
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let fingerprint = Sha256::digest(serde_json::to_vec(&input)?).to_vec();
        let name = self.tool.name();
        let skipped = {
            let mut state = self.state.lock().map_err(|error| {
                std::io::Error::other(format!("Tool call guard is unavailable: {error}"))
            })?;
            if state.seen.contains(&(name.clone(), fingerprint.clone())) {
                Some("Tool call skipped: identical arguments already produced a result. Use that result and answer the user.")
            } else if state.calls >= state.max_calls {
                Some("Tool call skipped: the per-turn tool-call limit was reached. Use the results already available and answer the user.")
            } else {
                state.seen.insert((name, fingerprint));
                state.calls += 1;
                None
            }
        };
        if let Some(message) = skipped {
            return Ok(message.to_owned());
        }
        self.tool.run(input).await
    }
}

pub struct SqliteBaseMemory {
    connection: Arc<StdMutex<Connection>>,
    thread_id: String,
    max_context_messages: usize,
}

impl SqliteBaseMemory {
    pub fn new(
        connection: Arc<StdMutex<Connection>>,
        thread_id: impl Into<String>,
        max_context_messages: usize,
    ) -> Result<Self, MemoryError> {
        connection
            .lock()
            .map_err(|error| MemoryError::Storage(error.to_string()))?
            .execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_langchain_agent_messages_thread
                    ON {MEMORY_TABLE}(thread_id, sequence);"
            ))
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(Self {
            connection,
            thread_id: thread_id.into(),
            max_context_messages,
        })
    }

    pub async fn matches_transcript(&self, expected: &[Message]) -> Result<bool, MemoryError> {
        let connection = self.lock_connection()?;
        let mut statement = connection
            .prepare(&format!(
                "SELECT message_json FROM {MEMORY_TABLE}
                 WHERE thread_id = ?1 ORDER BY sequence"
            ))
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let stored = statement
            .query_map([&self.thread_id], |row| row.get::<_, String>(0))
            .map_err(|error| MemoryError::Storage(error.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        drop(statement);
        let visible = stored
            .into_iter()
            .map(|serialized| serde_json::from_str::<Message>(&serialized))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|message| {
                message.message_type == MessageType::HumanMessage
                    || (message.message_type == MessageType::AIMessage
                        && message.tool_calls.is_none())
            })
            .collect::<Vec<_>>();
        Ok(visible.len() == expected.len()
            && visible.iter().zip(expected).all(|(actual, expected)| {
                actual.message_type == expected.message_type
                    && actual.content == expected.content
                    && actual.images == expected.images
            }))
    }

    fn lock_connection(&self) -> Result<std::sync::MutexGuard<'_, Connection>, MemoryError> {
        self.connection
            .lock()
            .map_err(|error| MemoryError::Storage(error.to_string()))
    }
}

#[async_trait]
impl BaseMemory for SqliteBaseMemory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        if self.max_context_messages == 0 {
            return Ok(Vec::new());
        }

        let connection = self.lock_connection()?;
        let limit = i64::try_from(self.max_context_messages).unwrap_or(i64::MAX);
        let mut statement = connection
            .prepare(&format!(
                "SELECT sequence, message_json FROM {MEMORY_TABLE}
                 WHERE thread_id = ?1 ORDER BY sequence DESC LIMIT ?2"
            ))
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let recent = statement
            .query_map(params![self.thread_id, limit], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| MemoryError::Storage(error.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        drop(statement);

        let Some((first_sequence, first_message)) = recent.last() else {
            return Ok(Vec::new());
        };
        let first_message: Message = serde_json::from_str(first_message)?;
        let mut serialized = Vec::new();
        if first_message.message_type != MessageType::HumanMessage {
            let preceding_human_sequence = {
                let mut statement = connection
                    .prepare(&format!(
                        "SELECT sequence, message_json FROM {MEMORY_TABLE}
                         WHERE thread_id = ?1 AND sequence < ?2
                         ORDER BY sequence DESC"
                    ))
                    .map_err(|error| MemoryError::Storage(error.to_string()))?;
                let mut rows = statement
                    .query(params![self.thread_id, first_sequence])
                    .map_err(|error| MemoryError::Storage(error.to_string()))?;
                let mut human_sequence = None;
                while let Some(row) = rows
                    .next()
                    .map_err(|error| MemoryError::Storage(error.to_string()))?
                {
                    let sequence = row
                        .get::<_, i64>(0)
                        .map_err(|error| MemoryError::Storage(error.to_string()))?;
                    let message: Message = serde_json::from_str(
                        &row.get::<_, String>(1)
                            .map_err(|error| MemoryError::Storage(error.to_string()))?,
                    )?;
                    if message.message_type == MessageType::HumanMessage {
                        human_sequence = Some(sequence);
                        break;
                    }
                }
                human_sequence
            };

            if let Some(human_sequence) = preceding_human_sequence {
                let mut statement = connection
                    .prepare(&format!(
                        "SELECT message_json FROM {MEMORY_TABLE}
                         WHERE thread_id = ?1 AND sequence >= ?2 AND sequence < ?3
                         ORDER BY sequence"
                    ))
                    .map_err(|error| MemoryError::Storage(error.to_string()))?;
                serialized.extend(
                    statement
                        .query_map(
                            params![self.thread_id, human_sequence, first_sequence],
                            |row| row.get::<_, String>(0),
                        )
                        .map_err(|error| MemoryError::Storage(error.to_string()))?
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| MemoryError::Storage(error.to_string()))?,
                );
            }
        }
        serialized.extend(recent.into_iter().rev().map(|(_, message)| message));
        serialized
            .into_iter()
            .map(|message| serde_json::from_str(&message).map_err(MemoryError::from))
            .collect()
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        let serialized = serde_json::to_string(&message)?;
        self.lock_connection()?
            .execute(
                &format!("INSERT INTO {MEMORY_TABLE}(thread_id, message_json) VALUES (?1, ?2)"),
                params![self.thread_id, serialized],
            )
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.lock_connection()?
            .execute(
                &format!("DELETE FROM {MEMORY_TABLE} WHERE thread_id = ?1"),
                params![self.thread_id],
            )
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(())
    }
}

#[async_trait]
pub trait McpToolClient: Send + Sync {
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
    ) -> Result<Value, McpConnectionError>;
}

#[derive(Debug)]
pub struct McpConnectionError(String);

impl std::fmt::Display for McpConnectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for McpConnectionError {}

impl McpConnectionError {
    fn new(message: String) -> Self {
        Self(message)
    }
}

pub struct McpConnection {
    _service: Arc<RunningService<RoleClient, ClientConfig>>,
    tools: Vec<Arc<dyn Tool>>,
}

impl McpConnection {
    pub async fn connect_stdio(
        namespace: &str,
        command: Command,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let transport = TokioChildProcess::new(command)?;
        Self::connect(transport, namespace).await
    }

    pub async fn connect_streamable_http(
        namespace: &str,
        uri: &str,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let transport = StreamableHttpClientTransport::from_uri(uri);
        Self::connect(transport, namespace).await
    }

    #[cfg(feature = "mcp-demo")]
    pub async fn connect_streamable_http_with_headers(
        namespace: &str,
        uri: &str,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let config = StreamableHttpClientTransportConfig::with_uri(uri).custom_headers(headers);
        let transport = StreamableHttpClientTransport::from_config(config);
        Self::connect(transport, namespace).await
    }

    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    async fn connect<T, E, A>(
        transport: T,
        namespace: &str,
    ) -> Result<Self, Box<dyn Error + Send + Sync>>
    where
        T: rmcp::transport::IntoTransport<RoleClient, E, A>,
        E: Error + Send + Sync + 'static,
    {
        let config = ClientConfig::new(
            ClientCapabilities::default(),
            Implementation::new("multimodal-local-chatbot", env!("CARGO_PKG_VERSION")),
        );
        let service = Arc::new(
            config
                .serve_with_lifecycle(transport, ClientLifecycleMode::Initialize)
                .await?,
        );
        let peer = service.peer().clone();
        let definitions = peer.list_tools(None).await?.tools;
        let client: Arc<dyn McpToolClient> = Arc::new(RmcpToolClient {
            peer,
            _service: service.clone(),
        });
        let tools = definitions
            .into_iter()
            .map(|definition| {
                let schema = serde_json::to_value(definition.input_schema.as_ref())?;
                let name = if namespace.is_empty() {
                    definition.name.to_string()
                } else {
                    format!("{namespace}__{}", definition.name)
                };
                Ok(Arc::new(McpTool::new(
                    name,
                    definition.name.to_string(),
                    definition.description.unwrap_or_default().to_string(),
                    schema,
                    client.clone(),
                )) as Arc<dyn Tool>)
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()?;
        Ok(Self {
            _service: service,
            tools,
        })
    }
}

struct RmcpToolClient {
    peer: Peer<RoleClient>,
    _service: Arc<RunningService<RoleClient, ClientConfig>>,
}

#[async_trait]
impl McpToolClient for RmcpToolClient {
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
    ) -> Result<Value, McpConnectionError> {
        let arguments = arguments.as_object().cloned().ok_or_else(|| {
            McpConnectionError::new("MCP tool arguments must be a JSON object".into())
        })?;
        let result = self
            .peer
            .call_tool(CallToolRequestParams::new(tool_name.to_owned()).with_arguments(arguments))
            .await
            .map_err(|error| McpConnectionError::new(error.to_string()))?;
        serde_json::to_value(result).map_err(|error| McpConnectionError::new(error.to_string()))
    }
}

pub struct McpTool {
    name: String,
    remote_name: String,
    description: String,
    input_schema: Value,
    client: Arc<dyn McpToolClient>,
}

impl McpTool {
    pub fn new(
        name: impl Into<String>,
        remote_name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        client: Arc<dyn McpToolClient>,
    ) -> Self {
        Self {
            name: name.into(),
            remote_name: remote_name.into(),
            description: description.into(),
            input_schema,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn parameters(&self) -> Value {
        self.input_schema.clone()
    }

    async fn parse_input(&self, input: &str) -> Value {
        parse_tool_input(input)
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        if !input.is_object() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MCP tool arguments must be a JSON object",
            )
            .into());
        }
        let result = self
            .client
            .call_tool(&self.remote_name, input)
            .await
            .map_err(|error| Box::new(error) as Box<dyn Error>)?;
        Ok(serde_json::to_string(&result)?)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        pin::Pin,
        sync::{Arc, Mutex as StdMutex},
    };

    use async_trait::async_trait;
    use futures_util::{stream, Stream, StreamExt};
    use langchain_rust::{
        agent::{AgentExecutor, AgentStreamEvent, OpenAiToolAgentBuilder},
        language_models::{GenerateResult, LLMError, LLMEvent, LLMEventStream, TokenUsage},
        output_parsers::{ResponseSchema, StructuredOutputParser, TypedOutputParser},
        prompt_args,
        schemas::{memory::BaseMemory, Message, StreamData, ToolCall},
        tools::Tool,
    };
    use serde_json::{json, Value};
    use tokio::sync::Mutex;

    use super::{AgentToolCallGuard, McpConnectionError, McpTool, McpToolClient, SqliteBaseMemory};

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct WeatherArguments {
        city: String,
    }

    struct TestMcpServer;

    #[rmcp::tool_router]
    impl TestMcpServer {
        #[rmcp::tool(description = "Return a deterministic forecast for a city.")]
        async fn get_weather(
            &self,
            rmcp::handler::server::wrapper::Parameters(arguments): rmcp::handler::server::wrapper::Parameters<
                WeatherArguments,
            >,
        ) -> String {
            format!("{}: sunny", arguments.city)
        }
    }

    #[rmcp::tool_handler]
    impl rmcp::handler::server::ServerHandler for TestMcpServer {}

    #[derive(Default)]
    struct RecordingMcpClient {
        calls: StdMutex<Vec<(String, Value)>>,
    }

    #[async_trait]
    impl McpToolClient for RecordingMcpClient {
        async fn call_tool(
            &self,
            tool_name: &str,
            arguments: Value,
        ) -> Result<Value, McpConnectionError> {
            self.calls
                .lock()
                .unwrap()
                .push((tool_name.to_string(), arguments.clone()));
            Ok(json!({"city": arguments["city"], "forecast": "sunny"}))
        }
    }

    struct FailingMcpClient;

    #[async_trait]
    impl McpToolClient for FailingMcpClient {
        async fn call_tool(
            &self,
            _tool_name: &str,
            _arguments: Value,
        ) -> Result<Value, McpConnectionError> {
            Err(McpConnectionError::new("offline".into()))
        }
    }

    #[tokio::test]
    async fn sqlite_memory_persists_messages_and_keeps_complete_tool_turns() {
        let path = std::env::temp_dir().join(format!(
            "langchain-memory-{}-{}.sqlite3",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let connection = Arc::new(StdMutex::new(rusqlite::Connection::open(&path).unwrap()));
        let mut memory = SqliteBaseMemory::new(connection.clone(), "timeline-1", 4).unwrap();
        memory
            .add_message(Message::new_human_message("weather in Lima"))
            .await
            .unwrap();
        memory
            .add_message(
                Message::new_ai_message("")
                    .with_tool_calls(json!([{"id":"call-1","type":"function"}])),
            )
            .await
            .unwrap();
        memory
            .add_message(Message::new_tool_message("sunny", "call-1"))
            .await
            .unwrap();
        memory
            .add_message(Message::new_ai_message("It is sunny."))
            .await
            .unwrap();
        drop(memory);
        drop(connection);

        let reopened = Arc::new(StdMutex::new(rusqlite::Connection::open(&path).unwrap()));
        let memory = SqliteBaseMemory::new(reopened, "timeline-1", 2).unwrap();
        let restored = memory.messages().await.unwrap();
        assert_eq!(restored.len(), 4);
        assert_eq!(restored[0].content, "weather in Lima");
        assert!(restored[1].tool_calls.is_some());
        assert_eq!(restored[2].id.as_deref(), Some("call-1"));
        assert_eq!(restored[3].content, "It is sunny.");

        drop(memory);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sqlite_memory_prunes_old_messages_to_a_sliding_window() {
        let connection = Arc::new(StdMutex::new(
            rusqlite::Connection::open_in_memory().unwrap(),
        ));
        let mut memory = SqliteBaseMemory::new(connection, "timeline-2", 2).unwrap();
        memory
            .add_message(Message::new_human_message("old question"))
            .await
            .unwrap();
        memory
            .add_message(Message::new_ai_message("old answer"))
            .await
            .unwrap();
        memory
            .add_message(Message::new_human_message("new question"))
            .await
            .unwrap();
        memory
            .add_message(Message::new_ai_message("new answer"))
            .await
            .unwrap();

        let context = memory.messages().await.unwrap();
        assert_eq!(
            context
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>(),
            vec!["new question", "new answer"]
        );
    }

    #[tokio::test]
    async fn sqlite_memory_matches_the_visible_transcript_without_dropping_tool_turns() {
        let connection = Arc::new(StdMutex::new(
            rusqlite::Connection::open_in_memory().unwrap(),
        ));
        let mut memory = SqliteBaseMemory::new(connection, "transcript-check", 20).unwrap();
        let user = Message::new_human_message("Find the schedule");
        let answer = Message::new_ai_message("The meeting is at noon.");
        memory.add_message(user.clone()).await.unwrap();
        memory
            .add_message(
                Message::new_ai_message("")
                    .with_tool_calls(json!([{"id":"call-1","type":"function"}])),
            )
            .await
            .unwrap();
        memory
            .add_message(Message::new_tool_message("12:00", "call-1"))
            .await
            .unwrap();
        memory.add_message(answer.clone()).await.unwrap();

        assert!(memory
            .matches_transcript(&[user.clone(), answer])
            .await
            .unwrap());
        assert!(!memory
            .matches_transcript(&[Message::new_human_message("Changed"), user])
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn mcp_tool_preserves_schema_arguments_and_structured_result() {
        let client = Arc::new(RecordingMcpClient::default());
        let tool = McpTool::new(
            "mcp_weather_lookup",
            "weather_lookup",
            "Get the weather for a city.",
            json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
            client.clone(),
        );

        assert_eq!(tool.parameters()["required"], json!(["city"]));
        let output = tool.call(r#"{"city":"Lima"}"#).await.unwrap();
        let calls = client.calls.lock().unwrap();
        assert_eq!(calls[0].0, "weather_lookup");
        assert_eq!(calls[0].1, json!({"city": "Lima"}));
        assert_eq!(
            serde_json::from_str::<Value>(&output).unwrap(),
            json!({"city": "Lima", "forecast": "sunny"})
        );
    }

    #[tokio::test]
    async fn agent_tool_call_guard_skips_repeated_and_excess_calls() {
        let client = Arc::new(RecordingMcpClient::default());
        let tool: Arc<dyn Tool> = Arc::new(McpTool::new(
            "mcp_weather_lookup",
            "weather_lookup",
            "Look up a city's weather.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            client.clone(),
        ));
        let guard = AgentToolCallGuard::new(1);
        let guarded = guard.wrap(tool);

        let first = guarded.call(r#"{"city":"Lima"}"#).await.unwrap();
        let duplicate = guarded.call(r#"{"city":"Lima"}"#).await.unwrap();
        let excess = guarded.call(r#"{"city":"Quito"}"#).await.unwrap();

        assert!(first.contains("sunny"));
        assert!(duplicate.contains("identical arguments"));
        assert!(excess.contains("per-turn tool-call limit"));
        assert_eq!(client.calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn mcp_tool_preserves_connection_failure_type_for_connector_management() {
        let tool = McpTool::new(
            "mcp_test",
            "test",
            "Exercise a failed MCP transport.",
            json!({"type": "object"}),
            Arc::new(FailingMcpClient),
        );

        let error = tool.run(json!({})).await.unwrap_err();

        assert!(error.downcast_ref::<McpConnectionError>().is_some());
    }

    #[tokio::test]
    async fn rmcp_connection_discovers_and_invokes_a_server_tool() {
        use rmcp::service::ServiceExt;

        let (client_transport, server_transport) = tokio::io::duplex(64 * 1024);
        let server_task = tokio::spawn(async move {
            TestMcpServer
                .serve(server_transport)
                .await
                .unwrap()
                .waiting()
                .await
        });
        let connection = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            super::McpConnection::connect(client_transport, "weather"),
        )
        .await
        .expect("MCP initialize and tool discovery timed out")
        .unwrap();
        let tools = connection.tools();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "weather__get_weather");
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tools[0].run(json!({"city": "Lima"})),
        )
        .await
        .expect("MCP tool call timed out")
        .unwrap();
        let output: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(output["content"][0]["text"], "Lima: sunny");

        drop(tools);
        drop(connection);
        tokio::time::timeout(std::time::Duration::from_secs(10), server_task)
            .await
            .expect("MCP server shutdown timed out")
            .unwrap()
            .unwrap();
    }

    #[test]
    fn structured_output_parser_checks_required_fields() {
        let parser = StructuredOutputParser::new([
            ResponseSchema::new("answer", "The response."),
            ResponseSchema::new("confidence", "A confidence score."),
        ]);
        let output = parser
            .parse_output(r#"{"answer":"42","confidence":0.9}"#)
            .unwrap();
        assert_eq!(output["answer"], "42");
        assert_eq!(output["confidence"], 0.9);
        assert!(parser.parse_output(r#"{"answer":"42"}"#).is_err());
    }

    #[tokio::test]
    async fn agent_streams_tool_events_then_final_answer_chunks() {
        let client = Arc::new(RecordingMcpClient::default());
        let weather = Arc::new(McpTool::new(
            "weather_lookup",
            "weather_lookup",
            "Look up a city's weather.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            client,
        ));
        let tools: Vec<Arc<dyn Tool>> = vec![weather];
        let llm = ScriptedLlm::new([
            ScriptedResponse::tool_call(ToolCall::new(
                "call-1",
                "weather_lookup",
                r#"{"city":"Lima"}"#,
            )),
            ScriptedResponse::text("It is ", "sunny."),
        ]);
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&tools)
            .build(llm)
            .unwrap();
        let memory = Arc::new(Mutex::new(
            SqliteBaseMemory::new(
                Arc::new(StdMutex::new(
                    rusqlite::Connection::open_in_memory().unwrap(),
                )),
                "agent-thread",
                16,
            )
            .unwrap(),
        ));
        let executor = AgentExecutor::from_agent(agent).with_memory(memory.clone());
        let mut events = executor.stream_events(prompt_args! {"input" => "weather in Lima"});
        let mut collected = Vec::new();
        while let Some(event) = events.next().await {
            collected.push(event.unwrap());
        }

        let tool_start = collected
            .iter()
            .position(|event| matches!(event, AgentStreamEvent::ToolStart { .. }))
            .unwrap();
        let tool_end = collected
            .iter()
            .position(|event| matches!(event, AgentStreamEvent::ToolEnd { .. }))
            .unwrap();
        let first_text = collected
            .iter()
            .position(|event| matches!(event, AgentStreamEvent::Text(_)))
            .unwrap();
        assert!(tool_start < tool_end && tool_end < first_text);
        let streamed_text = collected
            .iter()
            .filter_map(|event| match event {
                AgentStreamEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(streamed_text, "It is sunny.");
        assert!(matches!(
            collected.last(),
            Some(AgentStreamEvent::Finish { .. })
        ));

        let saved = memory.lock().await.messages().await.unwrap();
        assert_eq!(saved.len(), 4);
        assert!(saved[1].tool_calls.is_some());
        assert_eq!(
            saved[2].message_type,
            langchain_rust::schemas::MessageType::ToolMessage
        );
        assert_eq!(saved[3].content, "It is sunny.");
    }

    #[tokio::test]
    async fn agent_stream_can_emit_planning_text_before_a_tool_call() {
        let client = Arc::new(RecordingMcpClient::default());
        let weather = Arc::new(McpTool::new(
            "weather_lookup",
            "weather_lookup",
            "Look up a city's weather.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            client,
        ));
        let tools: Vec<Arc<dyn Tool>> = vec![weather];
        let llm = ScriptedLlm::new([
            ScriptedResponse::tool_call_with_preamble(
                ToolCall::new("call-1", "weather_lookup", r#"{"city":"Lima"}"#),
                "I'll check the weather.",
            ),
            ScriptedResponse::text("It is ", "sunny."),
        ]);
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&tools)
            .build(llm)
            .unwrap();
        let executor = AgentExecutor::from_agent(agent);
        let mut events = executor.stream_events(prompt_args! {"input" => "weather in Lima"});
        let mut collected = Vec::new();
        while let Some(event) = events.next().await {
            collected.push(event.unwrap());
        }

        let planning_text = collected
            .iter()
            .position(|event| matches!(event, AgentStreamEvent::Text(text) if text == "I'll check the weather."))
            .unwrap();
        let tool_start = collected
            .iter()
            .position(|event| matches!(event, AgentStreamEvent::ToolStart { .. }))
            .unwrap();
        assert!(planning_text < tool_start);
    }

    enum ScriptedResponse {
        ToolCall(ToolCall),
        ToolCallWithPreamble(ToolCall, &'static str),
        Text(&'static str, &'static str),
    }

    impl ScriptedResponse {
        fn tool_call(call: ToolCall) -> Self {
            Self::ToolCall(call)
        }

        fn tool_call_with_preamble(call: ToolCall, preamble: &'static str) -> Self {
            Self::ToolCallWithPreamble(call, preamble)
        }

        fn text(first: &'static str, second: &'static str) -> Self {
            Self::Text(first, second)
        }
    }

    #[derive(Clone)]
    struct ScriptedLlm {
        responses: Arc<StdMutex<VecDeque<ScriptedResponse>>>,
    }

    impl ScriptedLlm {
        fn new(responses: impl IntoIterator<Item = ScriptedResponse>) -> Self {
            Self {
                responses: Arc::new(StdMutex::new(responses.into_iter().collect())),
            }
        }

        fn next_result(&self) -> GenerateResult {
            match self.responses.lock().unwrap().pop_front().unwrap() {
                ScriptedResponse::ToolCall(call)
                | ScriptedResponse::ToolCallWithPreamble(call, _) => {
                    GenerateResult::default().with_tool_calls(vec![call])
                }
                ScriptedResponse::Text(first, second) => GenerateResult {
                    generation: format!("{first}{second}"),
                    tokens: Some(TokenUsage::new(12, 4)),
                    ..Default::default()
                },
            }
        }
    }

    #[async_trait]
    impl langchain_rust::language_models::llm::LLM for ScriptedLlm {
        async fn generate(&self, _messages: &[Message]) -> Result<GenerateResult, LLMError> {
            Ok(self.next_result())
        }

        async fn stream(
            &self,
            _messages: &[Message],
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError>
        {
            panic!("the agent test uses generate_stream")
        }

        fn generate_stream<'a>(&'a self, _messages: &'a [Message]) -> LLMEventStream<'a> {
            let response = self.responses.lock().unwrap().pop_front().unwrap();
            let (result, preamble) = match response {
                ScriptedResponse::ToolCall(call) => {
                    (GenerateResult::default().with_tool_calls(vec![call]), None)
                }
                ScriptedResponse::ToolCallWithPreamble(call, preamble) => (
                    GenerateResult::default().with_tool_calls(vec![call]),
                    Some(preamble),
                ),
                ScriptedResponse::Text(first, second) => (
                    GenerateResult {
                        generation: format!("{first}{second}"),
                        tokens: Some(TokenUsage::new(12, 4)),
                        ..Default::default()
                    },
                    None,
                ),
            };
            let mut events = Vec::new();
            if let Some(preamble) = preamble {
                events.push(LLMEvent::Text(preamble.to_string()));
            }
            if result.has_tool_calls() {
                events.push(LLMEvent::Done(result));
            } else {
                let full = result.generation.clone();
                let split = full.find("sunny.").unwrap();
                events.push(LLMEvent::Text(full[..split].to_string()));
                events.push(LLMEvent::Text(full[split..].to_string()));
                events.push(LLMEvent::Done(result));
            }
            Box::pin(stream::iter(events.into_iter().map(Ok)))
        }
    }
}
