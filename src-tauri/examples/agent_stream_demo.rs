use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;

const SERVER_PORT: u16 = 18081;
const SERVER_URL: &str = "http://127.0.0.1:18081";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let server_path = repo_root
        .join("vendor")
        .join("llama.cpp")
        .join("windows-x86_64")
        .join("llama-server.exe");
    let model_path = repo_root
        .join("models")
        .join("LFM3-Omni")
        .join("LFM3-Omni-2B-Q4_K_M.gguf");

    if !server_path.exists() {
        return Err(format!("Missing llama-server binary: {}", server_path.display()).into());
    }
    if !model_path.exists() {
        return Err(format!("Missing model file: {}", model_path.display()).into());
    }

    let mut child = Command::new(&server_path)
        .arg("--model")
        .arg(&model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(SERVER_PORT.to_string())
        .arg("--no-webui")
        .arg("--threads")
        .arg("4")
        .arg("--threads-batch")
        .arg("4")
        .arg("--parallel")
        .arg("1")
        .arg("--ctx-size")
        .arg("4096")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    if let Err(err) = wait_for_server().await {
        let _ = child.kill();
        return Err(err.into());
    }

    let client = Client::new();
    let model_id = get_model_id(&client).await?;

    println!("\n=== agent-style tool call stage ===");
    let tool_call = stream_completion(
        &client,
        &model_id,
        vec![json!({ "role": "user", "content": "Use the lookup_weather tool for Seattle." })],
        Some(json!([
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Return short weather information for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        },
                        "required": ["city"]
                    }
                }
            }
        ])),
        |delta: &str| {
            print!("[tool-step-stream] {delta}");
            std::io::stdout().flush().unwrap();
        },
    )
    .await?;

    println!("\n\nTool arguments captured: {tool_call}\n");

    println!("=== final answer stage ===");
    let messages = vec![
        json!({ "role": "user", "content": "Use the lookup_weather tool for Seattle." }),
        json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"Seattle\"}"
                }
            }]
        }),
        json!({
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "lookup_weather",
            "content": "{\"city\":\"Seattle\",\"temperature_c\":18,\"conditions\":\"cloudy\"}"
        }),
    ];

    let final_answer = stream_completion(&client, &model_id, messages, None, |delta: &str| {
        print!("{delta}");
        std::io::stdout().flush().unwrap();
    })
    .await?;

    println!("\n\nFinal answer assembled: {final_answer}\n");

    let _ = child.kill();
    Ok(())
}

async fn wait_for_server() -> Result<(), String> {
    let client = Client::new();
    for _ in 0..180 {
        match client.get(&format!("{SERVER_URL}/health")).send().await {
            Ok(response) if response.status().is_success() => return Ok(()),
            _ => sleep(Duration::from_secs(1)),
        }
    }
    Err(format!("Timed out waiting for llama.cpp at {SERVER_URL}"))
}

async fn get_model_id(client: &Client) -> Result<String, Box<dyn Error>> {
    let value: Value = client
        .get(&format!("{SERVER_URL}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let model = value["data"][0]["id"]
        .as_str()
        .ok_or("No model id returned by llama.cpp")?;
    Ok(model.to_string())
}

async fn stream_completion<F>(
    client: &Client,
    model_id: &str,
    messages: Vec<Value>,
    tools: Option<Value>,
    mut on_delta: F,
) -> Result<String, Box<dyn Error>>
where
    F: FnMut(&str),
{
    let mut payload = json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true }
    });

    if let Some(tools) = tools {
        payload["tools"] = tools;
        payload["tool_choice"] = json!("auto");
    }

    let response = client
        .post(&format!("{SERVER_URL}/v1/chat/completions"))
        .json(&payload)
        .send()
        .await?;

    let mut text = String::new();
    let mut tool_calls = String::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text_block = String::from_utf8_lossy(&chunk);
        for line in text_block.lines() {
            let raw = line.trim();
            if !raw.starts_with("data:") {
                continue;
            }
            let payload = raw.strip_prefix("data:").unwrap().trim();
            if payload == "[DONE]" {
                break;
            }
            let event: Value = match serde_json::from_str(payload) {
                Ok(value) => value,
                Err(_) => continue,
            };

            let choice = match event
                .get("choices")
                .and_then(|v| v.as_array())
                .and_then(|v| v.first())
            {
                Some(choice) => choice,
                None => continue,
            };

            let delta = choice.get("delta").cloned().unwrap_or(Value::Null);

            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    text.push_str(content);
                    on_delta(content);
                }
            }

            if let Some(tool_calls_value) = delta.get("tool_calls") {
                tool_calls.push_str(&tool_calls_value.to_string());
                on_delta(&format!("\n[tool_call] {}\n", tool_calls_value));
            }
        }
    }

    if !tool_calls.is_empty() {
        return Ok(tool_calls);
    }

    Ok(text)
}
