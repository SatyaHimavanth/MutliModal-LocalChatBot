use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::time::Duration;

const SERVER_URL: &str = "http://127.0.0.1:8080";
const EMBEDDING_DIMENSIONS: usize = 768;
const CHUNK_CHARS: usize = 900;
const CHUNK_OVERLAP: usize = 120;

#[derive(Clone)]
pub struct ChunkToEmbed {
    pub id: i64,
    pub document_id: i64,
    pub content: String,
    pub media_format: Option<String>,
    pub media_payload: Option<Vec<u8>>,
}

#[derive(Clone, serde::Serialize)]
pub struct RetrievedSource {
    pub filename: String,
    pub excerpt: String,
    pub distance: f64,
}

pub fn pending_chunks(
    db: &mut Connection,
    collection_id: i64,
) -> Result<Vec<ChunkToEmbed>, String> {
    // Replace legacy whole-file audio chunks once. Long recordings previously exceeded llama.cpp's
    // default 512-token physical batch; the new representation uses short overlapping segments.
    db.execute(
        "DELETE FROM chunk_vectors WHERE rowid IN (SELECT c.id FROM document_chunks c JOIN documents d ON d.id = c.document_id WHERE d.extension IN ('wav', 'mp3', 'flac') AND c.content LIKE 'Audio file:%' AND EXISTS (SELECT 1 FROM collection_documents cd WHERE cd.document_id = d.id AND cd.collection_id = ?1))",
        [collection_id],
    )
    .map_err(|error| format!("Could not refresh existing audio embeddings: {error}"))?;
    db.execute(
        "DELETE FROM document_chunks WHERE content LIKE 'Audio file:%' AND document_id IN (SELECT d.id FROM documents d WHERE d.extension IN ('wav', 'mp3', 'flac') AND EXISTS (SELECT 1 FROM collection_documents cd WHERE cd.document_id = d.id AND cd.collection_id = ?1))",
        [collection_id],
    )
    .map_err(|error| format!("Could not refresh existing audio chunks: {error}"))?;

    let unchunked = {
        let mut statement = db.prepare("SELECT d.id, d.extracted_text FROM documents d WHERE EXISTS (SELECT 1 FROM collection_documents cd WHERE cd.document_id = d.id AND cd.collection_id = ?1) AND NOT EXISTS (SELECT 1 FROM document_chunks c WHERE c.document_id = d.id)").map_err(|e| e.to_string())?;
        let rows = statement
            .query_map([collection_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };
    let tx = db.transaction().map_err(|e| e.to_string())?;
    for (document_id, text) in unchunked {
        let media: Option<(String, Vec<u8>)> = tx.query_row("SELECT extension, media_payload FROM documents WHERE id = ?1 AND media_payload IS NOT NULL", [document_id], |row| Ok((row.get(0)?, row.get(1)?))).ok();
        if let Some((format, payload)) = media {
            if matches!(format.as_str(), "wav" | "mp3" | "flac") {
                let segments = crate::audio::split_for_embedding(&payload, &format)?;
                for (ordinal, segment) in segments.into_iter().enumerate() {
                    let content = format!(
                        "Audio segment {} ({}–{} seconds)",
                        ordinal + 1,
                        segment.start_seconds,
                        segment.end_seconds
                    );
                    tx.execute(
                        "INSERT OR IGNORE INTO document_chunks(document_id, ordinal, content, modality, media_format, media_payload) VALUES (?1, ?2, ?3, 'audio', 'audio/wav', ?4)",
                        params![document_id, ordinal as i64, content, segment.wav],
                    )
                    .map_err(|error| format!("Could not save audio segment: {error}"))?;
                }
                continue;
            }
            let mime = match format.as_str() {
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "bmp" => "image/bmp",
                "gif" => "image/gif",
                "tga" => "image/x-tga",
                "wav" => "audio/wav",
                "mp3" => "audio/mpeg",
                "flac" => "audio/flac",
                _ => continue,
            };
            tx.execute("INSERT OR IGNORE INTO document_chunks(document_id, ordinal, content, modality, media_format, media_payload) VALUES (?1, 0, ?2, 'image', ?3, ?4)", params![document_id, format!("Image file: {}", format), mime, payload]).map_err(|e| e.to_string())?;
        } else {
            for (ordinal, content) in split_text(&text).into_iter().enumerate() {
                tx.execute("INSERT OR IGNORE INTO document_chunks(document_id, ordinal, content) VALUES (?1, ?2, ?3)", params![document_id, ordinal as i64, content]).map_err(|e| e.to_string())?;
            }
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    let mut statement = db.prepare("SELECT c.id, c.document_id, c.content, c.media_format, c.media_payload FROM document_chunks c JOIN collection_documents cd ON cd.document_id = c.document_id LEFT JOIN chunk_vectors v ON v.rowid = c.id WHERE cd.collection_id = ?1 AND v.rowid IS NULL ORDER BY c.document_id, c.id").map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([collection_id], |row| {
            Ok(ChunkToEmbed {
                id: row.get(0)?,
                document_id: row.get(1)?,
                content: row.get(2)?,
                media_format: row.get(3)?,
                media_payload: row.get(4)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

pub async fn embed_chunks_with_progress(
    chunks: &[ChunkToEmbed],
    mut on_progress: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>, String> {
    let mut output = Vec::with_capacity(chunks.len());
    let mut text_indices = Vec::new();
    let texts = chunks
        .iter()
        .enumerate()
        .filter_map(|(index, chunk)| {
            if chunk.media_payload.is_none() {
                text_indices.push(index);
                Some(chunk.content.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let text_vectors = embed_texts_with_progress(&texts, |count| on_progress(count)).await?;
    let mut by_index = std::collections::HashMap::new();
    for (index, vector) in text_indices.into_iter().zip(text_vectors) {
        by_index.insert(index, vector);
    }
    for (index, chunk) in chunks.iter().enumerate() {
        let Some(payload) = &chunk.media_payload else {
            output.push(by_index.remove(&index).ok_or("Missing text vector")?);
            continue;
        };
        if payload.len() > 8 * 1024 * 1024 {
            return Err("Media payload exceeds the 8 MB embedding limit.".into());
        }
        let encoded = base64_encode(payload);
        let content = match chunk.media_format.as_deref() {
            Some("image/png") | Some("image/jpeg") | Some("image/bmp") | Some("image/gif")
            | Some("image/x-tga") => {
                json!({"type":"image_url", "image_url":{"url":format!("data:{};base64,{}", chunk.media_format.as_deref().unwrap_or("image/png"), encoded)}})
            }
            Some("audio/wav") | Some("audio/mpeg") | Some("audio/flac") => {
                let format = match chunk.media_format.as_deref() {
                    Some("audio/mpeg") => "mp3",
                    Some("audio/flac") => "flac",
                    _ => "wav",
                };
                json!({"type":"input_audio", "input_audio":{"data":encoded,"format":format}})
            }
            _ => return Err("Unsupported media embedding format.".into()),
        };
        let vector = embed_single_content(content).await?;
        output.push(vector);
        on_progress(1);
    }
    Ok(output)
}

async fn embed_single_content(content: Value) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|e| e.to_string())?;
    let models: Value = client
        .get(format!("{SERVER_URL}/v1/models"))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|_| "The local model service is not responding.".to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let model = model_id(&models)
        .ok_or_else(|| "EmbeddingGemma was not found in the local model router.".to_string())?;
    let response = client
        .post(format!("{SERVER_URL}/v1/embeddings"))
        .json(&json!({"model":model,"input":[{"content":[content]}],"encoding_format":"float"}))
        .send()
        .await
        .map_err(|error| format!("Could not request media embedding: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| body.chars().take(500).collect());
        return Err(format!(
            "The local server could not embed this media segment (HTTP {status}): {detail}"
        ));
    }
    let response: Value = response.json().await.map_err(|error| error.to_string())?;
    let row = response
        .pointer("/data/0")
        .ok_or_else(|| "Media embedding response did not contain a vector.".to_string())?;
    parse_vector(row)
}

pub(crate) fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let value = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | chunk.get(2).copied().unwrap_or(0) as u32;
        output.push(TABLE[((value >> 18) & 63) as usize] as char);
        output.push(TABLE[((value >> 12) & 63) as usize] as char);
        output.push(if chunk.len() > 1 {
            TABLE[((value >> 6) & 63) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            TABLE[(value & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

pub async fn embed_texts(texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    embed_texts_with_progress(texts, |_| {}).await
}

async fn embed_texts_with_progress(
    texts: &[String],
    mut on_progress: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|e| e.to_string())?;
    let models: Value = client
        .get(format!("{SERVER_URL}/v1/models"))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|_| "The local model service is not responding.".to_string())?
        .error_for_status()
        .map_err(|e| format!("Could not read available models: {e}"))?
        .json()
        .await
        .map_err(|e| format!("The model service returned invalid model information: {e}"))?;
    let model = model_id(&models)
        .ok_or_else(|| "EmbeddingGemma was not found in the local model router.".to_string())?;
    let mut vectors = Vec::with_capacity(texts.len());
    // Keep requests small to limit peak memory on managed 2–3 GB devices.
    for batch in texts.chunks(4) {
        let input: Vec<Value> = batch
            .iter()
            .map(|text| json!({"content": [{"type": "text", "text": text}]}))
            .collect();
        let response: Value = client
            .post(format!("{SERVER_URL}/v1/embeddings"))
            .json(&json!({"model": model, "input": input, "encoding_format": "float"}))
            .send()
            .await
            .map_err(|e| format!("Could not request document embeddings: {e}"))?
            .error_for_status()
            .map_err(|e| format!("The embedding model could not process this batch: {e}"))?
            .json()
            .await
            .map_err(|e| format!("The embedding model returned invalid vectors: {e}"))?;
        let rows = response
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| "The embedding response did not contain vectors.".to_string())?;
        if rows.len() != batch.len() {
            return Err("The embedding response returned an unexpected vector count.".into());
        }
        for row in rows {
            vectors.push(parse_vector(row)?);
        }
        on_progress(batch.len());
    }
    Ok(vectors)
}

pub fn save_vectors(
    db: &mut Connection,
    chunks: &[ChunkToEmbed],
    vectors: &[Vec<f32>],
) -> Result<(), String> {
    if chunks.len() != vectors.len() {
        return Err("Embedding vector count mismatch.".into());
    }
    let tx = db.transaction().map_err(|e| e.to_string())?;
    for (chunk, vector) in chunks.iter().zip(vectors) {
        tx.execute(
            "INSERT OR REPLACE INTO chunk_vectors(rowid, embedding) VALUES (?1, ?2)",
            params![chunk.id, vector_bytes(vector)],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())
}

pub fn search(
    db: &Connection,
    vector: &[f32],
    query: &str,
    collection_ids: &[i64],
) -> Result<Vec<RetrievedSource>, String> {
    if collection_ids.is_empty() {
        return Ok(Vec::new());
    }
    const VECTOR_CANDIDATE_LIMIT: usize = 160;
    const KEYWORD_CANDIDATE_LIMIT: usize = 60;
    const RRF_K: f64 = 60.0;
    let selected = serde_json::to_string(collection_ids).map_err(|e| e.to_string())?;
    let mut ranked = std::collections::HashMap::<i64, (RetrievedSource, f64)>::new();
    let candidates = {
        let mut statement = db.prepare("SELECT v.rowid, v.distance, c.content, d.filename, d.id FROM chunk_vectors v JOIN document_chunks c ON c.id = v.rowid JOIN documents d ON d.id = c.document_id WHERE v.embedding MATCH ?1 AND v.k = ?2 ORDER BY v.distance").map_err(|e| e.to_string())?;
        let rows = statement
            .query_map(
                params![vector_bytes(vector), VECTOR_CANDIDATE_LIMIT as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                },
            )
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };
    let mut selected_vector_rank = 0usize;
    for (chunk_id, distance, content, filename, document_id) in candidates {
        let in_collections: bool = db
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM collection_documents WHERE document_id = ?1 AND collection_id IN (SELECT value FROM json_each(?2)))",
                params![document_id, selected],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        if !in_collections {
            continue;
        }
        ranked.insert(
            chunk_id,
            (
                RetrievedSource {
                    filename,
                    excerpt: content.chars().take(420).collect(),
                    distance,
                },
                1.0 / (RRF_K + selected_vector_rank as f64 + 1.0),
            ),
        );
        selected_vector_rank += 1;
    }

    if let Some(fts_query) = keyword_query(query) {
        let keyword_candidates = {
            let mut statement = db.prepare(
                "SELECT s.rowid, c.content, d.filename FROM chunk_search s JOIN document_chunks c ON c.id = s.rowid JOIN documents d ON d.id = c.document_id WHERE chunk_search MATCH ?1 AND EXISTS (SELECT 1 FROM collection_documents cd WHERE cd.document_id = d.id AND cd.collection_id IN (SELECT value FROM json_each(?2))) ORDER BY bm25(chunk_search) LIMIT ?3",
            ).map_err(|e| e.to_string())?;
            let rows = statement
                .query_map(
                    params![fts_query, selected, KEYWORD_CANDIDATE_LIMIT as i64],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )
                .map_err(|e| e.to_string())?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        };
        for (rank, (chunk_id, content, filename)) in keyword_candidates.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
            ranked
                .entry(chunk_id)
                .and_modify(|(_, score)| *score += contribution)
                .or_insert_with(|| {
                    (
                        RetrievedSource {
                            filename,
                            excerpt: content.chars().take(420).collect(),
                            distance: 0.0,
                        },
                        contribution,
                    )
                });
        }
    }

    let mut results = ranked.into_values().collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.filename.cmp(&right.0.filename))
    });
    Ok(results
        .into_iter()
        .take(6)
        .map(|(source, _)| source)
        .collect())
}

fn keyword_query(query: &str) -> Option<String> {
    let terms = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.chars().count() > 1)
        .take(24)
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

pub fn has_indexed_documents(db: &Connection, collection_ids: &[i64]) -> Result<bool, String> {
    let encoded = serde_json::to_string(collection_ids).map_err(|e| e.to_string())?;
    db.query_row(
        "SELECT EXISTS(SELECT 1 FROM collection_documents cd JOIN document_chunks c ON c.document_id = cd.document_id JOIN chunk_vectors v ON v.rowid = c.id WHERE cd.collection_id IN (SELECT value FROM json_each(?1)))",
        [encoded], |row| row.get(0),
    ).map_err(|e| e.to_string())
}

fn model_id(value: &Value) -> Option<String> {
    value
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|item| item.get("id")?.as_str())
        .find(|id| id.to_ascii_lowercase().contains("embeddinggemma"))
        .map(str::to_owned)
}

fn parse_vector(row: &Value) -> Result<Vec<f32>, String> {
    let values = row
        .pointer("/embedding")
        .and_then(Value::as_array)
        .ok_or_else(|| "The embedding response did not contain a vector.".to_string())?;
    if values.len() != EMBEDDING_DIMENSIONS {
        return Err(format!(
            "Expected {EMBEDDING_DIMENSIONS} embedding values, received {}.",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| {
            let number = value
                .as_f64()
                .ok_or_else(|| "The embedding contains a non-numeric value.".to_string())?
                as f32;
            if !number.is_finite() {
                return Err("The embedding contains a non-finite value.".into());
            }
            Ok(number)
        })
        .collect()
}

fn vector_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn split_text(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + CHUNK_CHARS).min(chars.len());
        let mut boundary = end;
        if end < chars.len() {
            if let Some(offset) = chars[start..end]
                .iter()
                .rposition(|ch| matches!(ch, '\n' | '.' | '!' | '?' | ' '))
            {
                if offset > CHUNK_CHARS / 2 {
                    boundary = start + offset + 1;
                }
            }
        }
        let chunk = chars[start..boundary]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        if boundary == chars.len() {
            break;
        }
        start = boundary.saturating_sub(CHUNK_OVERLAP);
    }
    chunks
}
