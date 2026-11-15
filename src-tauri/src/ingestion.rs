use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use calamine::{open_workbook_auto, Reader};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 100 * 1024 * 1024;
const MAX_EXTRACTED_CHARS: usize = 1_500_000;
const MAX_MEDIA_BYTES: u64 = 8 * 1024 * 1024;
pub const CHAT_MEDIA_TOTAL_LIMIT: u64 = 16 * 1024 * 1024;
pub const CHAT_ATTACHMENT_TOKEN_LIMIT: usize = 5_000;
const PARSER_VERSION: &str = "multimodal-v1";

#[derive(Serialize)]
pub struct Collection {
    pub id: i64,
    pub name: String,
    pub document_count: u64,
}

#[derive(Serialize)]
pub struct CollectionDocument {
    pub document_id: i64,
    pub filename: String,
    pub byte_size: u64,
    pub extension: String,
    pub chunk_count: u64,
    pub indexed_chunks: u64,
    pub paths: Vec<String>,
}

#[derive(Serialize)]
pub struct CollectionDocumentContent {
    pub filename: String,
    pub extension: String,
    pub extracted_text: String,
    pub has_media_payload: bool,
}

#[derive(Serialize)]
pub struct ImportResult {
    pub filename: String,
    pub file_index: usize,
    pub collection_id: i64,
    pub document_id: Option<i64>,
    pub byte_size: u64,
    pub extracted_chars: usize,
    pub reused_existing: bool,
    pub indexing_error: Option<String>,
    pub error: Option<String>,
    pub retry_token: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatAttachment {
    pub filename: String,
    pub path: String,
    pub byte_size: u64,
    pub estimated_tokens: usize,
    pub content: String,
    #[serde(default)]
    pub media_mime: Option<String>,
    #[serde(default)]
    pub media_format: Option<String>,
    #[serde(default)]
    pub media_data_base64: Option<String>,
}

#[derive(Serialize)]
pub struct PreparedChatAttachment {
    pub filename: String,
    pub byte_size: u64,
    pub estimated_tokens: usize,
    pub content: Option<String>,
    pub over_limit: bool,
    pub error: Option<String>,
    pub media_mime: Option<String>,
    pub media_format: Option<String>,
    pub media_data_base64: Option<String>,
}

pub fn prepare_chat_files(paths: Vec<String>) -> Vec<PreparedChatAttachment> {
    paths
        .into_iter()
        .map(|raw_path| {
            let path = PathBuf::from(&raw_path);
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Attached file")
                .to_string();
            let result = (|| {
                let path = fs::canonicalize(&path)
                    .map_err(|error| format!("Could not access file: {error}"))?;
                if !path.is_file() {
                    return Err("Only individual files can be attached.".to_string());
                }
                let byte_size = fs::metadata(&path)
                    .map_err(|error| error.to_string())?
                    .len();
                if byte_size == 0 || byte_size > MAX_FILE_BYTES {
                    return Err("Chat attachments must be between 1 byte and 50 MB.".into());
                }
                let extension = path
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if let Some((mime, format)) = media_type(&extension) {
                    if byte_size > MAX_MEDIA_BYTES {
                        return Err("Image and audio attachments are limited to 8 MB each.".into());
                    }
                    let bytes = fs::read(&path).map_err(|error| error.to_string())?;
                    return Ok((
                        byte_size,
                        0,
                        String::new(),
                        Some(mime.to_string()),
                        Some(format.to_string()),
                        Some(crate::retrieval::base64_encode(&bytes)),
                    ));
                }
                let text = extract_text(&path, &extension)?;
                if text.trim().is_empty() {
                    return Err("No text could be extracted from this file.".into());
                }
                if text.chars().count() > MAX_EXTRACTED_CHARS {
                    return Err("Extracted text exceeds the 1.5 million character limit.".into());
                }
                let estimated_tokens = text.chars().count().div_ceil(3);
                Ok((byte_size, estimated_tokens, text, None, None, None))
            })();
            match result {
                Ok((
                    byte_size,
                    estimated_tokens,
                    text,
                    media_mime,
                    media_format,
                    media_data_base64,
                )) => {
                    let over_limit = estimated_tokens > CHAT_ATTACHMENT_TOKEN_LIMIT;
                    PreparedChatAttachment {
                        filename,
                        byte_size,
                        estimated_tokens,
                        content: (!over_limit).then_some(text),
                        over_limit,
                        error: None,
                        media_mime,
                        media_format,
                        media_data_base64,
                    }
                }
                Err(error) => PreparedChatAttachment {
                    filename,
                    byte_size: 0,
                    estimated_tokens: 0,
                    content: None,
                    over_limit: false,
                    error: Some(error),
                    media_mime: None,
                    media_format: None,
                    media_data_base64: None,
                },
            }
        })
        .collect()
}

fn media_type(extension: &str) -> Option<(&'static str, &'static str)> {
    match extension {
        "png" => Some(("image/png", "png")),
        "jpg" | "jpeg" => Some(("image/jpeg", "jpeg")),
        "bmp" => Some(("image/bmp", "bmp")),
        "gif" => Some(("image/gif", "gif")),
        "tga" => Some(("image/x-tga", "tga")),
        "wav" => Some(("audio/wav", "wav")),
        "mp3" => Some(("audio/mpeg", "mp3")),
        "flac" => Some(("audio/flac", "flac")),
        _ => None,
    }
}

pub fn collection_document_content(
    db: &Connection,
    collection_id: i64,
    document_id: i64,
) -> Result<CollectionDocumentContent, String> {
    db.query_row(
        "SELECT d.filename, d.extension, d.extracted_text, d.media_payload IS NOT NULL FROM documents d JOIN collection_documents cd ON cd.document_id = d.id WHERE cd.collection_id = ?1 AND d.id = ?2",
        params![collection_id, document_id],
        |row| Ok(CollectionDocumentContent { filename: row.get(0)?, extension: row.get(1)?, extracted_text: row.get(2)?, has_media_payload: row.get(3)? }),
    ).map_err(|error| format!("Could not load extracted document content: {error}"))
}

pub fn list_collections(db: &Connection) -> Result<Vec<Collection>, String> {
    let mut statement = db
        .prepare(
            "SELECT c.id, c.name, COUNT(cd.document_id)
             FROM collections c
             LEFT JOIN collection_documents cd ON cd.collection_id = c.id
             GROUP BY c.id
             ORDER BY c.name COLLATE NOCASE",
        )
        .map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(Collection {
                id: row.get(0)?,
                name: row.get(1)?,
                document_count: row.get::<_, i64>(2)? as u64,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

pub fn list_collection_documents(
    db: &Connection,
    collection_id: i64,
) -> Result<Vec<CollectionDocument>, String> {
    let mut statement = db.prepare(
        "SELECT d.id, d.filename, d.byte_size, d.extension, COUNT(c.id), COUNT(v.rowid) FROM documents d JOIN collection_documents cd ON cd.document_id = d.id LEFT JOIN document_chunks c ON c.document_id = d.id LEFT JOIN chunk_vectors v ON v.rowid = c.id WHERE cd.collection_id = ?1 GROUP BY d.id ORDER BY d.filename COLLATE NOCASE",
    ).map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([collection_id], |row| {
            Ok(CollectionDocument {
                document_id: row.get(0)?,
                filename: row.get(1)?,
                byte_size: row.get::<_, i64>(2)?.max(0) as u64,
                extension: row.get(3)?,
                chunk_count: row.get::<_, i64>(4)?.max(0) as u64,
                indexed_chunks: row.get::<_, i64>(5)?.max(0) as u64,
                paths: Vec::new(),
            })
        })
        .map_err(|error| error.to_string())?;
    let mut documents = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    for document in &mut documents {
        let mut paths = db.prepare("SELECT path FROM document_paths WHERE document_id = ?1 ORDER BY path COLLATE NOCASE")
            .map_err(|error| error.to_string())?;
        document.paths = paths
            .query_map([document.document_id], |row| row.get(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
    }
    Ok(documents)
}

pub fn remove_collection_documents(
    db: &mut Connection,
    collection_id: i64,
    document_ids: &[i64],
) -> Result<usize, String> {
    if document_ids.is_empty() || document_ids.len() > 100 {
        return Err("Select between 1 and 100 files to remove at a time.".into());
    }
    let transaction = db.transaction().map_err(|error| error.to_string())?;
    let mut removed = 0;
    for document_id in document_ids {
        let associated: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM collection_documents WHERE collection_id = ?1 AND document_id = ?2)",
            params![collection_id, document_id], |row| row.get(0),
        ).map_err(|error| error.to_string())?;
        if !associated {
            return Err(
                "A selected file is no longer in this collection. Refresh the list and try again."
                    .into(),
            );
        }
        transaction
            .execute(
                "DELETE FROM collection_documents WHERE collection_id = ?1 AND document_id = ?2",
                params![collection_id, document_id],
            )
            .map_err(|error| error.to_string())?;
        let used_elsewhere: bool = transaction
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM collection_documents WHERE document_id = ?1)",
                [document_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if !used_elsewhere {
            transaction.execute("DELETE FROM chunk_vectors WHERE rowid IN (SELECT id FROM document_chunks WHERE document_id = ?1)", [document_id])
                .map_err(|error| format!("Could not remove the file's search vectors: {error}"))?;
            transaction
                .execute("DELETE FROM documents WHERE id = ?1", [document_id])
                .map_err(|error| {
                    format!("Could not remove the file and its extracted content: {error}")
                })?;
        }
        removed += 1;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(removed)
}

pub fn create_collection(db: &Connection, name: &str) -> Result<Collection, String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 50 {
        return Err("Collection names must be between 1 and 50 characters.".into());
    }
    db.execute(
        "INSERT INTO collections(name, created_at) VALUES (?1, ?2)",
        params![name, unix_time()],
    )
    .map_err(|error| {
        if error.to_string().contains("UNIQUE constraint failed") {
            "A collection with that name already exists.".to_string()
        } else {
            error.to_string()
        }
    })?;
    Ok(Collection {
        id: db.last_insert_rowid(),
        name: name.to_string(),
        document_count: 0,
    })
}

pub fn rename_collection(db: &Connection, id: i64, name: &str) -> Result<Collection, String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 50 {
        return Err("Collection names must be between 1 and 50 characters.".into());
    }
    db.execute(
        "UPDATE collections SET name = ?1 WHERE id = ?2",
        params![name, id],
    )
    .map_err(|error| {
        if error.to_string().contains("UNIQUE constraint failed") {
            "A collection with that name already exists.".to_string()
        } else {
            error.to_string()
        }
    })?;
    if db.changes() == 0 {
        return Err("That collection no longer exists.".into());
    }
    list_collections(db)?
        .into_iter()
        .find(|collection| collection.id == id)
        .ok_or_else(|| "That collection no longer exists.".to_string())
}

pub fn delete_collection(db: &Connection, id: i64) -> Result<(), String> {
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM collections", [], |row| row.get(0))
        .map_err(|error| error.to_string())?;
    if count <= 1 {
        return Err(
            "Keep at least one collection. Create another collection before deleting this one."
                .into(),
        );
    }
    db.execute("DELETE FROM collections WHERE id = ?1", [id])
        .map_err(|error| error.to_string())?;
    if db.changes() == 0 {
        return Err("That collection no longer exists.".into());
    }
    Ok(())
}

pub fn import_files_with_progress(
    db: &mut Connection,
    paths: Vec<String>,
    collection_id: i64,
    mut on_file: impl FnMut(usize, usize, usize, &str, Option<&str>),
) -> Result<Vec<ImportResult>, String> {
    if paths.is_empty() || paths.len() > 20 {
        return Err("Drop between 1 and 20 files at a time.".into());
    }
    let total_files = paths.len();
    let collection_exists: bool = db
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM collections WHERE id = ?1)",
            [collection_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    if !collection_exists {
        return Err("Choose an existing collection before importing files.".into());
    }

    let mut total_bytes = 0_u64;
    let mut results = Vec::with_capacity(paths.len());
    for (file_index, path) in paths.into_iter().enumerate() {
        let path = PathBuf::from(path);
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Unsupported file")
            .to_string();
        let processing_filename = filename.clone();
        on_file(file_index, total_files, file_index, &filename, None);
        let result = (|| {
            let path = fs::canonicalize(&path)
                .map_err(|error| format!("Could not access file: {error}"))?;
            if !path.is_file() {
                return Err("Only individual files can be imported.".to_string());
            }
            let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
            let byte_size = metadata.len();
            if byte_size == 0 {
                return Err("The file is empty.".to_string());
            }
            if byte_size > MAX_FILE_BYTES {
                return Err("Files are limited to 50 MB in this import build.".to_string());
            }
            total_bytes = total_bytes.saturating_add(byte_size);
            if total_bytes > MAX_TOTAL_BYTES {
                return Err("A batch is limited to 100 MB total.".to_string());
            }
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(
                extension.as_str(),
                "pdf"
                    | "docx"
                    | "pptx"
                    | "xlsx"
                    | "csv"
                    | "txt"
                    | "md"
                    | "png"
                    | "jpg"
                    | "jpeg"
                    | "bmp"
                    | "gif"
                    | "tga"
                    | "wav"
                    | "mp3"
                    | "flac"
            ) {
                return Err(format!(
                    "The .{extension} format is not supported by this importer yet."
                ));
            }
            let is_media = media_type(&extension).is_some();
            if is_media && byte_size > MAX_MEDIA_BYTES {
                return Err("Image and audio files are limited to 8 MB each in this build.".into());
            }
            let source_path = path.to_string_lossy().into_owned();
            let hash = hash_file(&path)?;

            let existing: Option<(i64, String, String)> = db
                .query_row(
                    "SELECT id, parser_version, extracted_text FROM documents WHERE sha256 = ?1",
                    [&hash],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()
                .map_err(|e| e.to_string())?;
            if let Some((document_id, parser_version, text)) = existing {
                if parser_version == PARSER_VERSION {
                    attach_document(db, document_id, collection_id, &source_path)?;
                    return Ok(ImportResult {
                        filename: processing_filename.clone(),
                        file_index,
                        collection_id,
                        document_id: Some(document_id),
                        byte_size,
                        extracted_chars: text.chars().count(),
                        reused_existing: true,
                        indexing_error: None,
                        error: None,
                        retry_token: None,
                    });
                }
                let refreshed_text = extract_text(&path, &extension)?;
                update_existing_document(
                    db,
                    document_id,
                    collection_id,
                    &source_path,
                    &refreshed_text,
                    &extension,
                )?;
                return Ok(ImportResult {
                    filename: processing_filename.clone(),
                    file_index,
                    collection_id,
                    document_id: Some(document_id),
                    byte_size,
                    extracted_chars: refreshed_text.chars().count(),
                    reused_existing: true,
                    indexing_error: None,
                    error: None,
                    retry_token: None,
                });
            }

            let text = if is_media {
                format!(
                    "{} file: {}",
                    if extension == "wav" || extension == "mp3" || extension == "flac" {
                        "Audio"
                    } else {
                        "Image"
                    },
                    processing_filename
                )
            } else {
                extract_text(&path, &extension)?
            };
            if text.trim().is_empty() {
                return Err(
                    "No text could be extracted. Scanned PDFs and OCR are not supported yet."
                        .to_string(),
                );
            }
            if text.chars().count() > MAX_EXTRACTED_CHARS {
                return Err("Extracted text is over the 1.5 million character limit.".to_string());
            }
            let transaction = db.transaction().map_err(|e| e.to_string())?;
            transaction
                .execute(
                    "INSERT INTO documents(sha256, filename, byte_size, extension, parser_version, extracted_text, created_at, media_payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![hash, processing_filename, byte_size as i64, extension, PARSER_VERSION, text, unix_time(), if is_media { Some(fs::read(&path).map_err(|e| e.to_string())?) } else { None }],
                )
                .map_err(|e| e.to_string())?;
            let document_id = transaction.last_insert_rowid();
            transaction
                .execute(
                    "INSERT INTO document_paths(document_id, path, added_at) VALUES (?1, ?2, ?3)",
                    params![document_id, source_path, unix_time()],
                )
                .map_err(|e| e.to_string())?;
            transaction.execute(
                "INSERT INTO collection_documents(collection_id, document_id, added_at) VALUES (?1, ?2, ?3)",
                params![collection_id, document_id, unix_time()],
            ).map_err(|e| e.to_string())?;
            transaction.commit().map_err(|e| e.to_string())?;
            Ok(ImportResult {
                filename: processing_filename,
                file_index,
                collection_id,
                document_id: Some(document_id),
                byte_size,
                extracted_chars: text.chars().count(),
                reused_existing: false,
                indexing_error: None,
                error: None,
                retry_token: None,
            })
        })();
        results.push(match result {
            Ok(result) => result,
            Err(error) => ImportResult {
                filename,
                file_index,
                collection_id,
                document_id: None,
                byte_size: 0,
                extracted_chars: 0,
                reused_existing: false,
                indexing_error: None,
                error: Some(error),
                retry_token: None,
            },
        });
        let processed_name = results
            .last()
            .map(|item| item.filename.as_str())
            .unwrap_or("Document");
        on_file(
            file_index + 1,
            total_files,
            file_index,
            processed_name,
            results.last().and_then(|result| result.error.as_deref()),
        );
    }
    Ok(results)
}

pub fn remove_failed_import(
    db: &mut Connection,
    collection_id: i64,
    document_id: i64,
) -> Result<(), String> {
    let transaction = db.transaction().map_err(|error| error.to_string())?;
    transaction
        .execute(
            "DELETE FROM collection_documents WHERE collection_id = ?1 AND document_id = ?2",
            params![collection_id, document_id],
        )
        .map_err(|error| error.to_string())?;
    let still_referenced: bool = transaction
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM collection_documents WHERE document_id = ?1)",
            [document_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if !still_referenced {
        transaction.execute(
            "DELETE FROM chunk_vectors WHERE rowid IN (SELECT id FROM document_chunks WHERE document_id = ?1)",
            [document_id],
        ).map_err(|error| error.to_string())?;
        transaction
            .execute("DELETE FROM documents WHERE id = ?1", [document_id])
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

fn attach_document(
    db: &mut Connection,
    document_id: i64,
    collection_id: i64,
    source_path: &str,
) -> Result<(), String> {
    let transaction = db.transaction().map_err(|e| e.to_string())?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO document_paths(document_id, path, added_at) VALUES (?1, ?2, ?3)",
            params![document_id, source_path, unix_time()],
        )
        .map_err(|e| e.to_string())?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO collection_documents(collection_id, document_id, added_at) VALUES (?1, ?2, ?3)",
            params![collection_id, document_id, unix_time()],
        )
        .map_err(|e| e.to_string())?;
    transaction.commit().map_err(|e| e.to_string())
}

fn update_existing_document(
    db: &mut Connection,
    document_id: i64,
    collection_id: i64,
    source_path: &str,
    extracted_text: &str,
    extension: &str,
) -> Result<(), String> {
    if extracted_text.trim().is_empty() || extracted_text.chars().count() > MAX_EXTRACTED_CHARS {
        return Err("The updated parser returned empty or oversized document text.".into());
    }
    let transaction = db.transaction().map_err(|e| e.to_string())?;
    transaction.execute(
        "DELETE FROM chunk_vectors WHERE rowid IN (SELECT id FROM document_chunks WHERE document_id = ?1)",
        [document_id],
    ).map_err(|e| e.to_string())?;
    transaction
        .execute(
            "DELETE FROM document_chunks WHERE document_id = ?1",
            [document_id],
        )
        .map_err(|e| e.to_string())?;
    transaction
        .execute(
            "UPDATE documents SET parser_version = ?1, extension = ?2, extracted_text = ?3 WHERE id = ?4",
            params![PARSER_VERSION, extension, extracted_text, document_id],
        )
        .map_err(|e| e.to_string())?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO document_paths(document_id, path, added_at) VALUES (?1, ?2, ?3)",
            params![document_id, source_path, unix_time()],
        )
        .map_err(|e| e.to_string())?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO collection_documents(collection_id, document_id, added_at) VALUES (?1, ?2, ?3)",
            params![collection_id, document_id, unix_time()],
        )
        .map_err(|e| e.to_string())?;
    transaction.commit().map_err(|e| e.to_string())
}

fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_text(path: &Path, extension: &str) -> Result<String, String> {
    let text = match extension {
        "pdf" => pdf_extract::extract_text(path)
            .map_err(|error| format!("PDF extraction failed: {error}"))?,
        "docx" => extract_office_xml(path, &["word/document.xml"])?,
        "pptx" => extract_presentation(path)?,
        "xlsx" => extract_spreadsheet(path)?,
        "csv" => extract_csv(path)?,
        "txt" | "md" => fs::read_to_string(path)
            .map_err(|error| format!("Could not read text file: {error}"))?,
        "png" | "jpg" | "jpeg" | "wav" => String::new(),
        other => {
            return Err(format!(
                "The .{other} format is not supported by this importer yet."
            ))
        }
    };
    Ok(text)
}

fn extract_csv(path: &Path) -> Result<String, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .map_err(|error| format!("CSV parsing failed: {error}"))?;
    let mut output = String::new();
    for row in reader.records() {
        let row = row.map_err(|error| format!("CSV parsing failed: {error}"))?;
        output.push_str(&row.iter().collect::<Vec<_>>().join("\t"));
        output.push('\n');
        check_text_limit(&output)?;
    }
    Ok(output)
}

fn extract_spreadsheet(path: &Path) -> Result<String, String> {
    let mut workbook =
        open_workbook_auto(path).map_err(|error| format!("Spreadsheet parsing failed: {error}"))?;
    let mut output = String::new();
    for (name, range) in workbook.worksheets() {
        output.push_str("Sheet: ");
        output.push_str(&name);
        output.push('\n');
        for row in range.rows() {
            for (index, value) in row.iter().enumerate() {
                if index > 0 {
                    output.push('\t');
                }
                output.push_str(&value.to_string());
            }
            output.push('\n');
            check_text_limit(&output)?;
        }
    }
    Ok(output)
}

fn extract_presentation(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| format!("PPTX container is invalid: {error}"))?;
    let mut slides = Vec::new();
    for index in 0..archive.len() {
        let entry = archive.by_index(index).map_err(|error| error.to_string())?;
        let name = entry.name().to_string();
        if let Some(number) = name
            .strip_prefix("ppt/slides/slide")
            .and_then(|value| value.strip_suffix(".xml"))
            .and_then(|value| value.parse::<u32>().ok())
        {
            slides.push((number, name));
        }
    }
    slides.sort_by_key(|(number, _)| *number);
    let names: Vec<String> = slides.into_iter().map(|(_, name)| name).collect();
    extract_office_xml(path, &names)
}

fn extract_office_xml<S: AsRef<str>>(path: &Path, entries: &[S]) -> Result<String, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| format!("Office document container is invalid: {error}"))?;
    let mut output = String::new();
    let mut total_uncompressed = 0_u64;
    for name in entries {
        let name = name.as_ref();
        let entry = archive
            .by_name(name)
            .map_err(|error| format!("Office document is missing {name}: {error}"))?;
        total_uncompressed = total_uncompressed.saturating_add(entry.size());
        if total_uncompressed > 20 * 1024 * 1024 {
            return Err("Expanded Office document text exceeds the 20 MB parser limit.".into());
        }
        let mut xml = Vec::with_capacity(entry.size().min(4 * 1024 * 1024) as usize);
        entry
            .take(20 * 1024 * 1024)
            .read_to_end(&mut xml)
            .map_err(|error| error.to_string())?;
        let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
        reader.config_mut().trim_text(true);
        let mut buffer = Vec::new();
        loop {
            match reader.read_event_into(&mut buffer) {
                Ok(quick_xml::events::Event::Text(event)) => {
                    let decoded = event.decode().map_err(|error| error.to_string())?;
                    let unescaped =
                        quick_xml::escape::unescape(&decoded).map_err(|error| error.to_string())?;
                    output.push_str(&unescaped);
                    output.push(' ');
                    check_text_limit(&output)?;
                }
                Ok(quick_xml::events::Event::End(event)) => {
                    if matches!(event.name().as_ref(), b"w:p" | b"a:p") {
                        output.push('\n');
                    }
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(error) => return Err(format!("Office XML parsing failed: {error}")),
                _ => {}
            }
            buffer.clear();
        }
    }
    Ok(output)
}

fn check_text_limit(text: &str) -> Result<(), String> {
    if text.chars().count() > MAX_EXTRACTED_CHARS {
        Err("Extracted text is over the 1.5 million character limit.".into())
    } else {
        Ok(())
    }
}

fn unix_time() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}
