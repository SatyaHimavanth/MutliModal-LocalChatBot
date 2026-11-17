use std::collections::HashMap;

use http::header::{HeaderName, HeaderValue};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};

const MAX_CONNECTORS: usize = 32;
const MAX_HEADERS: usize = 32;
const MAX_HEADER_VALUE_BYTES: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024;

#[derive(Clone, Deserialize)]
pub struct McpHeaderInput {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct StoredHeader {
    name: String,
    value: String,
}

#[derive(Clone)]
pub struct McpConnector {
    pub id: i64,
    pub name: String,
    pub endpoint: String,
    pub headers: HashMap<String, String>,
}

#[derive(Serialize)]
pub struct McpConnectorSummary {
    pub id: i64,
    pub name: String,
    pub endpoint: String,
    pub available: bool,
    pub tool_count: usize,
    pub last_checked_at: Option<i64>,
    pub last_error: Option<String>,
    pub has_custom_headers: bool,
    pub selected: bool,
}

pub fn ensure_schema(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS mcp_connectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                endpoint TEXT NOT NULL,
                headers_json TEXT NOT NULL DEFAULT '[]',
                available INTEGER NOT NULL DEFAULT 0,
                tool_count INTEGER NOT NULL DEFAULT 0,
                last_checked_at INTEGER,
                last_error TEXT
            );
            CREATE TABLE IF NOT EXISTS conversation_mcp_selection (
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                connector_id INTEGER NOT NULL REFERENCES mcp_connectors(id) ON DELETE CASCADE,
                PRIMARY KEY(conversation_id, connector_id)
            );",
        )
        .map_err(|error| format!("Could not initialize MCP demo storage: {error}"))
}

pub fn validate_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 80 || name.chars().any(char::is_control) {
        return Err("Connector names must contain 1–80 printable characters.".into());
    }
    Ok(name.to_owned())
}

pub fn validate_endpoint(endpoint: &str) -> Result<reqwest::Url, String> {
    let endpoint = endpoint.trim();
    let parsed =
        reqwest::Url::parse(endpoint).map_err(|_| "Enter a valid MCP endpoint URL.".to_string())?;
    if parsed.username() != ""
        || parsed.password().is_some()
        || parsed.fragment().is_some()
        || parsed.query().is_some()
    {
        return Err(
            "MCP endpoint URLs must not contain credentials, query parameters, or fragments; use custom headers for authentication."
                .into(),
        );
    }
    match parsed.scheme() {
        "https" => {}
        "http" if is_loopback_host(parsed.host_str().unwrap_or_default()) => {}
        _ => {
            return Err("MCP endpoints must use HTTPS. HTTP is allowed only for localhost.".into())
        }
    }
    if parsed.host_str().is_none() {
        return Err("Enter a valid MCP endpoint URL.".into());
    }
    Ok(parsed)
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|address| address.is_loopback())
        .unwrap_or(false)
}

pub fn parse_custom_headers(
    headers: &[McpHeaderInput],
) -> Result<HashMap<HeaderName, HeaderValue>, String> {
    if headers.len() > MAX_HEADERS {
        return Err(format!("Use no more than {MAX_HEADERS} custom headers."));
    }
    let mut parsed = HashMap::with_capacity(headers.len());
    let mut total_bytes = 0usize;
    for header in headers {
        let name = HeaderName::from_bytes(header.name.trim().as_bytes())
            .map_err(|_| format!("Invalid HTTP header name: {}.", header.name.trim()))?;
        if matches!(
            name.as_str(),
            "connection" | "content-length" | "host" | "transfer-encoding" | "upgrade"
        ) {
            return Err(format!(
                "The {} header cannot be customized.",
                name.as_str()
            ));
        }
        if header.value.len() > MAX_HEADER_VALUE_BYTES {
            return Err("Custom HTTP header values are limited to 8 KB each.".into());
        }
        let value = HeaderValue::from_str(header.value.trim())
            .map_err(|_| format!("Invalid value for the {} header.", name.as_str()))?;
        total_bytes = total_bytes.saturating_add(name.as_str().len() + value.as_bytes().len());
        if total_bytes > MAX_HEADER_BYTES {
            return Err("Custom HTTP headers are limited to 16 KB total.".into());
        }
        if parsed.insert(name.clone(), value).is_some() {
            return Err(format!(
                "The {} header was entered more than once.",
                name.as_str()
            ));
        }
    }
    Ok(parsed)
}

fn serialize_headers(headers: &[McpHeaderInput]) -> Result<String, String> {
    parse_custom_headers(headers)?;
    serde_json::to_string(
        &headers
            .iter()
            .map(|header| StoredHeader {
                name: header.name.trim().to_owned(),
                value: header.value.trim().to_owned(),
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|error| format!("Could not store MCP authentication headers: {error}"))
}

pub fn save_connector(
    connection: &Connection,
    id: Option<i64>,
    name: &str,
    endpoint: &str,
    headers: Option<&[McpHeaderInput]>,
) -> Result<i64, String> {
    let name = validate_name(name)?;
    let endpoint = validate_endpoint(endpoint)?.to_string();
    let headers_json = headers.map(serialize_headers).transpose()?;
    match id {
        Some(id) => {
            let changed = connection
                .execute(
                    "UPDATE mcp_connectors SET name = ?1, endpoint = ?2,
                        headers_json = COALESCE(?3, headers_json),
                        available = 0, tool_count = 0, last_checked_at = NULL, last_error = NULL
                     WHERE id = ?4",
                    params![name, endpoint, headers_json, id],
                )
                .map_err(|error| format!("Could not save the MCP connector: {error}"))?;
            if changed == 0 {
                return Err("The MCP connector no longer exists.".into());
            }
            Ok(id)
        }
        None => connection
            .query_row(
                "INSERT INTO mcp_connectors(name, endpoint, headers_json)
                 VALUES (?1, ?2, ?3) RETURNING id",
                params![name, endpoint, headers_json.unwrap_or_else(|| "[]".into())],
                |row| row.get(0),
            )
            .map_err(|error| format!("Could not add the MCP connector: {error}")),
    }
}

pub fn load_connector(connection: &Connection, id: i64) -> Result<McpConnector, String> {
    let (name, endpoint, headers_json): (String, String, String) = connection
        .query_row(
            "SELECT name, endpoint, headers_json FROM mcp_connectors WHERE id = ?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(|error| format!("Could not read the MCP connector: {error}"))?
        .ok_or_else(|| "The MCP connector no longer exists.".to_string())?;
    let stored: Vec<StoredHeader> = serde_json::from_str(&headers_json)
        .map_err(|error| format!("Stored MCP authentication headers are invalid: {error}"))?;
    let inputs: Vec<McpHeaderInput> = stored
        .into_iter()
        .map(|header| McpHeaderInput {
            name: header.name,
            value: header.value,
        })
        .collect();
    parse_custom_headers(&inputs)?;
    let headers = inputs
        .into_iter()
        .map(|header| (header.name, header.value))
        .collect();
    Ok(McpConnector {
        id,
        name,
        endpoint,
        headers,
    })
}

pub fn list_connectors(
    connection: &Connection,
    conversation_id: Option<i64>,
) -> Result<Vec<McpConnectorSummary>, String> {
    let mut statement = connection
        .prepare(
            "SELECT c.id, c.name, c.endpoint, c.available, c.tool_count,
                    c.last_checked_at, c.last_error, c.headers_json,
                    EXISTS(SELECT 1 FROM conversation_mcp_selection s
                           WHERE s.connector_id = c.id AND s.conversation_id = ?1)
             FROM mcp_connectors c ORDER BY c.name COLLATE NOCASE",
        )
        .map_err(|error| format!("Could not list MCP connectors: {error}"))?;
    let rows = statement
        .query_map([conversation_id], |row| {
            let headers_json: String = row.get(7)?;
            Ok(McpConnectorSummary {
                id: row.get(0)?,
                name: row.get(1)?,
                endpoint: row.get(2)?,
                available: row.get::<_, i64>(3)? != 0,
                tool_count: row.get::<_, i64>(4)?.max(0) as usize,
                last_checked_at: row.get(5)?,
                last_error: row.get(6)?,
                has_custom_headers: headers_json != "[]",
                selected: row.get(8)?,
            })
        })
        .map_err(|error| format!("Could not list MCP connectors: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Could not list MCP connectors: {error}"))?;
    Ok(rows)
}

pub fn record_connection_result(
    connection: &Connection,
    id: i64,
    result: Result<usize, String>,
) -> Result<(), String> {
    let now = unix_time();
    let (available, count, error) = match result {
        Ok(count) => (1, i64::try_from(count).unwrap_or(i64::MAX), None),
        Err(_) => (
            0,
            0,
            Some("Could not connect or discover tools. Check the endpoint, authentication, and network.".to_string()),
        ),
    };
    let changed = connection
        .execute(
            "UPDATE mcp_connectors SET available = ?1, tool_count = ?2,
                last_checked_at = ?3, last_error = ?4 WHERE id = ?5",
            params![available, count, now, error, id],
        )
        .map_err(|error| format!("Could not update MCP connection status: {error}"))?;
    if changed == 0 {
        return Err("The MCP connector no longer exists.".into());
    }
    Ok(())
}

pub fn delete_connector(connection: &Connection, id: i64) -> Result<(), String> {
    let changed = connection
        .execute("DELETE FROM mcp_connectors WHERE id = ?1", [id])
        .map_err(|error| format!("Could not delete the MCP connector: {error}"))?;
    if changed == 0 {
        return Err("The MCP connector no longer exists.".into());
    }
    Ok(())
}

pub fn set_conversation_selection(
    connection: &mut Connection,
    conversation_id: i64,
    connector_ids: &[i64],
) -> Result<(), String> {
    if connector_ids.len() > MAX_CONNECTORS {
        return Err(format!("Select no more than {MAX_CONNECTORS} connectors."));
    }
    let unique = connector_ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if unique.len() != connector_ids.len() {
        return Err("A connector was selected more than once.".into());
    }
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| format!("Could not update connector selection: {error}"))?;
    for id in connector_ids {
        let available: Option<i64> = transaction
            .query_row(
                "SELECT available FROM mcp_connectors WHERE id = ?1",
                [id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| format!("Could not validate connector selection: {error}"))?;
        if available != Some(1) {
            return Err("Unavailable connectors must be reconnected before enabling them.".into());
        }
    }
    transaction
        .execute(
            "DELETE FROM conversation_mcp_selection WHERE conversation_id = ?1",
            [conversation_id],
        )
        .map_err(|error| format!("Could not update connector selection: {error}"))?;
    for id in connector_ids {
        transaction
            .execute(
                "INSERT INTO conversation_mcp_selection(conversation_id, connector_id)
                 VALUES (?1, ?2)",
                params![conversation_id, id],
            )
            .map_err(|error| format!("Could not update connector selection: {error}"))?;
    }
    transaction
        .commit()
        .map_err(|error| format!("Could not update connector selection: {error}"))
}

pub fn disable_connector_selections(
    connection: &Connection,
    connector_id: i64,
) -> Result<(), String> {
    connection
        .execute(
            "DELETE FROM conversation_mcp_selection WHERE connector_id = ?1",
            [connector_id],
        )
        .map_err(|error| format!("Could not disable the MCP connector: {error}"))?;
    Ok(())
}

pub fn selected_connectors(
    connection: &Connection,
    conversation_id: i64,
    connector_ids: &[i64],
) -> Result<Vec<McpConnector>, String> {
    let summaries = list_connectors(connection, Some(conversation_id))?;
    let mut connectors = Vec::with_capacity(connector_ids.len());
    for id in connector_ids {
        let summary = summaries
            .iter()
            .find(|item| item.id == *id)
            .ok_or_else(|| "The MCP connector no longer exists.".to_string())?;
        if !summary.selected {
            return Err("Connector selection changed. Refresh the list and try again.".into());
        }
        if !summary.available {
            return Err(
                "An enabled MCP connector is unavailable. Reconnect it and try again.".into(),
            );
        }
        connectors.push(load_connector(connection, *id)?);
    }
    Ok(connectors)
}

fn unix_time() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::{
        delete_connector, disable_connector_selections, ensure_schema, list_connectors,
        parse_custom_headers, record_connection_result, save_connector, validate_endpoint,
        validate_name, McpHeaderInput,
    };
    use rusqlite::Connection;

    #[test]
    fn allows_https_and_loopback_http_endpoints() {
        assert!(validate_endpoint("https://mcp.example.com/mcp").is_ok());
        assert!(validate_endpoint("http://127.0.0.1:8000/mcp").is_ok());
        assert!(validate_endpoint("http://localhost:8000/mcp").is_ok());
    }

    #[test]
    fn rejects_remote_http_and_urls_with_embedded_credentials() {
        assert!(validate_endpoint("http://mcp.example.com/mcp").is_err());
        assert!(validate_endpoint("https://user:pass@mcp.example.com/mcp").is_err());
    }

    #[test]
    fn rejects_empty_or_oversized_connector_names() {
        assert!(validate_name("  ").is_err());
        assert!(validate_name(&"x".repeat(81)).is_err());
        assert_eq!(validate_name("  Calendar  ").unwrap(), "Calendar");
    }

    #[test]
    fn validates_custom_headers_before_transport_use() {
        let headers = |name: &str, value: &str| {
            vec![McpHeaderInput {
                name: name.into(),
                value: value.into(),
            }]
        };
        assert!(parse_custom_headers(&headers("x-api-key", "secret")).is_ok());
        assert!(parse_custom_headers(&headers("bad header", "value")).is_err());
        assert!(parse_custom_headers(&headers("host", "attacker.example")).is_err());
        assert!(parse_custom_headers(&headers("x-api-key", "one\r\ntwo")).is_err());
    }

    #[test]
    fn connector_status_never_returns_saved_header_values() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("CREATE TABLE conversations (id INTEGER PRIMARY KEY);")
            .unwrap();
        ensure_schema(&connection).unwrap();
        let headers = [McpHeaderInput {
            name: "x-api-key".into(),
            value: "private-token".into(),
        }];
        let id = save_connector(
            &connection,
            None,
            "Calendar",
            "https://mcp.example.com/mcp",
            Some(&headers),
        )
        .unwrap();
        record_connection_result(&connection, id, Ok(2)).unwrap();
        let summaries = list_connectors(&connection, None).unwrap();
        let json = serde_json::to_string(&summaries).unwrap();
        assert!(json.contains("has_custom_headers"));
        assert!(!json.contains("private-token"));
        assert_eq!(summaries[0].tool_count, 2);
        delete_connector(&connection, id).unwrap();
        assert!(list_connectors(&connection, None).unwrap().is_empty());
    }

    #[test]
    fn failed_connector_is_disabled_for_the_conversation() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE conversations (id INTEGER PRIMARY KEY);
                 INSERT INTO conversations(id) VALUES (7);",
            )
            .unwrap();
        ensure_schema(&connection).unwrap();
        let id = save_connector(
            &connection,
            None,
            "Calendar",
            "https://mcp.example.com/mcp",
            None,
        )
        .unwrap();
        record_connection_result(&connection, id, Ok(1)).unwrap();
        super::set_conversation_selection(&mut connection, 7, &[id]).unwrap();

        record_connection_result(&connection, id, Err("offline".into())).unwrap();
        disable_connector_selections(&connection, id).unwrap();

        let connector = list_connectors(&connection, Some(7)).unwrap().remove(0);
        assert!(!connector.available);
        assert!(!connector.selected);
    }
}
