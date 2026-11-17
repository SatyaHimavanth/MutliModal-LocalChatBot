use std::{fs, path::Path};

use rusqlite::{Connection, OptionalExtension};

const DATABASE_NAME: &str = "conversations.sqlite3";
const KEY_NAME: &str = "database-key.dpapi";
const INDEX_FINGERPRINT: &str =
    "media-v1|text-v1|chars-900-overlap-120|embeddinggemma-2-768|vectors-v1";

pub fn open(app_data_dir: &Path) -> Result<Connection, String> {
    // sqlite-vec is statically linked and registered before opening the encrypted DB.
    // Its initializer uses SQLite's extension ABI and is process-global.
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
    fs::create_dir_all(app_data_dir)
        .map_err(|error| format!("Could not create the local app data folder: {error}"))?;
    let database_path = app_data_dir.join(DATABASE_NAME);
    let protected_key_path = app_data_dir.join(KEY_NAME);
    let mut key = load_or_create_key(&protected_key_path, database_path.exists())?;
    let connection = open_encrypted_connection(&database_path, &mut key)?;
    let pragma_result = connection.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;",
    );
    pragma_result
        .map_err(|error| format!("Could not configure encrypted local storage: {error}"))?;

    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 title TEXT NOT NULL,
                 created_at INTEGER NOT NULL,
                 updated_at INTEGER NOT NULL,
                 parent_conversation_id INTEGER REFERENCES conversations(id),
                 fork_message_id INTEGER,
                 pinned INTEGER NOT NULL DEFAULT 0,
                 archived INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                 role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                 content TEXT NOT NULL,
                 created_at INTEGER NOT NULL,
                 model TEXT,
                 input_tokens INTEGER,
                 output_tokens INTEGER,
                 elapsed_ms INTEGER,
                 time_to_first_token_ms INTEGER,
                 input_tokens_per_second REAL,
                 output_tokens_per_second REAL,
                 attachments_json TEXT,
                 generation_status TEXT NOT NULL DEFAULT 'complete'
             );
             CREATE INDEX IF NOT EXISTS messages_conversation_order
                 ON messages(conversation_id, id);
             CREATE TABLE IF NOT EXISTS timeline_hidden_messages (
                 conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                 message_id INTEGER NOT NULL,
                 PRIMARY KEY(conversation_id, message_id)
             );
             CREATE TABLE IF NOT EXISTS collections (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                 created_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS documents (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 sha256 TEXT NOT NULL UNIQUE,
                 filename TEXT NOT NULL,
                 byte_size INTEGER NOT NULL,
                 extension TEXT NOT NULL,
                 parser_version TEXT NOT NULL,
                 extracted_text TEXT NOT NULL,
                 created_at INTEGER NOT NULL,
                 media_payload BLOB
             );
             CREATE TABLE IF NOT EXISTS document_paths (
                 document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                 path TEXT NOT NULL,
                 added_at INTEGER NOT NULL,
                 PRIMARY KEY(document_id, path)
             );
             CREATE TABLE IF NOT EXISTS collection_documents (
                 collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                 document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                 added_at INTEGER NOT NULL,
                 PRIMARY KEY(collection_id, document_id)
             );
             CREATE INDEX IF NOT EXISTS collection_documents_collection
                 ON collection_documents(collection_id, document_id);
             CREATE TABLE IF NOT EXISTS document_chunks (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                 ordinal INTEGER NOT NULL,
                 content TEXT NOT NULL,
                 UNIQUE(document_id, ordinal)
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(content, filename, tokenize = 'unicode61 remove_diacritics 2');
             CREATE TRIGGER IF NOT EXISTS document_chunks_search_insert AFTER INSERT ON document_chunks BEGIN
                 INSERT INTO chunk_search(rowid, content, filename)
                 SELECT NEW.id, NEW.content, d.filename FROM documents d WHERE d.id = NEW.document_id;
             END;
             CREATE TRIGGER IF NOT EXISTS document_chunks_search_delete AFTER DELETE ON document_chunks BEGIN
                 DELETE FROM chunk_search WHERE rowid = OLD.id;
             END;
             CREATE TRIGGER IF NOT EXISTS document_chunks_search_update AFTER UPDATE OF content ON document_chunks BEGIN
                 UPDATE chunk_search SET content = NEW.content WHERE rowid = NEW.id;
             END;
             CREATE TABLE IF NOT EXISTS message_collections (
                 message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                 collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                 PRIMARY KEY(message_id, collection_id)
             );
             CREATE TABLE IF NOT EXISTS message_selection (
                 message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                 collection_ids_json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS message_sources (
                 message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                 source_index INTEGER NOT NULL,
                 filename TEXT NOT NULL,
                 excerpt TEXT NOT NULL,
                 distance REAL NOT NULL,
                 PRIMARY KEY(message_id, source_index)
             );
             CREATE TABLE IF NOT EXISTS agent_tool_invocations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 assistant_message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                 sequence INTEGER NOT NULL,
                 tool_name TEXT NOT NULL,
                 connector_name TEXT NOT NULL,
                 status TEXT NOT NULL CHECK(status IN ('running', 'complete', 'failed', 'skipped', 'interrupted')),
                 summary TEXT NOT NULL DEFAULT '',
                 elapsed_ms INTEGER,
                 created_at INTEGER NOT NULL,
                 UNIQUE(assistant_message_id, sequence)
             );
             CREATE INDEX IF NOT EXISTS agent_tool_invocations_message_order
                 ON agent_tool_invocations(assistant_message_id, sequence);
             CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(embedding float[768]);
             CREATE TABLE IF NOT EXISTS index_metadata (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             INSERT OR IGNORE INTO collections(name, created_at)
                 VALUES ('General', CAST(strftime('%s', 'now') AS INTEGER));",
        )
        .map_err(|error| format!("Could not initialize local conversation history: {error}"))?;

    ensure_chunk_media_columns(&connection)?;
    ensure_document_media_column(&connection)?;
    let saved_fingerprint: Option<String> = connection
        .query_row(
            "SELECT value FROM index_metadata WHERE key = 'fingerprint'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| format!("Could not read the local search index version: {error}"))?;
    ensure_conversation_branch_columns(&connection)?;
    ensure_conversation_management_columns(&connection)?;
    ensure_message_timing_columns(&connection)?;
    ensure_message_attachments_column(&connection)?;
    ensure_message_generation_status_column(&connection)?;
    connection
        .execute(
            "UPDATE messages SET content = 'Generation was interrupted before the app closed.', generation_status = 'interrupted' WHERE role = 'assistant' AND generation_status = 'pending'",
            [],
        )
        .map_err(|error| format!("Could not recover interrupted chat history: {error}"))?;
    connection
        .execute(
            "UPDATE agent_tool_invocations SET status = 'interrupted', summary = 'Tool call was interrupted before the app closed.' WHERE status = 'running'",
            [],
        )
        .map_err(|error| format!("Could not recover interrupted tool activity: {error}"))?;
    connection.execute(
        "INSERT OR IGNORE INTO chunk_search(rowid, content, filename) SELECT c.id, c.content, d.filename FROM document_chunks c JOIN documents d ON d.id = c.document_id",
        [],
    ).map_err(|error| format!("Could not synchronize the keyword search index: {error}"))?;
    if saved_fingerprint
        .as_deref()
        .is_some_and(|value| value != INDEX_FINGERPRINT)
    {
        connection
            .execute_batch("DELETE FROM chunk_vectors; DELETE FROM document_chunks;")
            .map_err(|error| format!("Could not refresh the local search index: {error}"))?;
    }
    connection
        .execute(
            "INSERT OR REPLACE INTO index_metadata(key, value) VALUES ('fingerprint', ?1)",
            [INDEX_FINGERPRINT],
        )
        .map_err(|error| format!("Could not save the local search index version: {error}"))?;
    #[cfg(feature = "mcp-demo")]
    crate::mcp_demo::ensure_schema(&connection)?;
    Ok(connection)
}

pub fn open_agent_memory_connection(app_data_dir: &Path) -> Result<Connection, String> {
    let database_path = app_data_dir.join(DATABASE_NAME);
    let protected_key_path = app_data_dir.join(KEY_NAME);
    let mut key = load_or_create_key(&protected_key_path, true)?;
    open_encrypted_connection(&database_path, &mut key)
}

fn open_encrypted_connection(database_path: &Path, key: &mut [u8]) -> Result<Connection, String> {
    let connection = Connection::open(database_path)
        .map_err(|error| format!("Could not open the encrypted local database: {error}"))?;
    let mut hex_key = String::with_capacity(key.len() * 2);
    for byte in key.iter() {
        use std::fmt::Write;
        let _ = write!(hex_key, "{byte:02x}");
    }
    key.fill(0);
    let mut pragma = format!("PRAGMA key = \"x'{hex_key}'\";");
    let pragma_result = connection.execute_batch(&pragma);
    // Best-effort clearing of the temporary textual copy of the data key.
    unsafe { pragma.as_bytes_mut().fill(0) };
    unsafe { hex_key.as_bytes_mut().fill(0) };
    pragma_result.map_err(|error| format!("Could not unlock the local database: {error}"))?;
    let cipher_version: String = connection
        .query_row("PRAGMA cipher_version", [], |row| row.get(0))
        .map_err(|error| format!("SQLCipher is unavailable: {error}"))?;
    if cipher_version.is_empty() {
        return Err("SQLCipher did not report an active encryption version.".into());
    }
    Ok(connection)
}

fn ensure_message_attachments_column(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(messages)")
        .map_err(|error| error.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    if !columns.iter().any(|column| column == "attachments_json") {
        connection
            .execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT", [])
            .map_err(|error| format!("Could not update chat attachment storage: {error}"))?;
    }
    Ok(())
}

fn ensure_message_generation_status_column(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(messages)")
        .map_err(|error| error.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    if !columns.iter().any(|column| column == "generation_status") {
        connection
            .execute(
                "ALTER TABLE messages ADD COLUMN generation_status TEXT NOT NULL DEFAULT 'complete'",
                [],
            )
            .map_err(|error| format!("Could not add chat generation state: {error}"))?;
    }
    Ok(())
}

fn ensure_message_timing_columns(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(messages)")
        .map_err(|error| error.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    if !columns
        .iter()
        .any(|column| column == "time_to_first_token_ms")
    {
        connection
            .execute(
                "ALTER TABLE messages ADD COLUMN time_to_first_token_ms INTEGER",
                [],
            )
            .map_err(|error| format!("Could not update saved response metrics: {error}"))?;
    }
    Ok(())
}

fn ensure_chunk_media_columns(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(document_chunks)")
        .map_err(|e| e.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    drop(statement);
    for (name, sql_type) in [
        ("modality", "TEXT NOT NULL DEFAULT 'text'"),
        ("media_format", "TEXT"),
        ("media_payload", "BLOB"),
    ] {
        if !columns.iter().any(|column| column == name) {
            connection
                .execute(
                    &format!("ALTER TABLE document_chunks ADD COLUMN {name} {sql_type}"),
                    [],
                )
                .map_err(|e| format!("Could not update local document storage: {e}"))?;
        }
    }
    Ok(())
}

fn ensure_document_media_column(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(documents)")
        .map_err(|e| e.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if !columns.iter().any(|column| column == "media_payload") {
        connection
            .execute("ALTER TABLE documents ADD COLUMN media_payload BLOB", [])
            .map_err(|e| format!("Could not update local document storage: {e}"))?;
    }
    Ok(())
}

fn ensure_conversation_branch_columns(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(conversations)")
        .map_err(|error| error.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    if !columns
        .iter()
        .any(|column| column == "parent_conversation_id")
    {
        connection.execute(
            "ALTER TABLE conversations ADD COLUMN parent_conversation_id INTEGER REFERENCES conversations(id)",
            [],
        ).map_err(|error| format!("Could not update local conversation storage: {error}"))?;
    }
    if !columns.iter().any(|column| column == "fork_message_id") {
        connection
            .execute(
                "ALTER TABLE conversations ADD COLUMN fork_message_id INTEGER",
                [],
            )
            .map_err(|error| format!("Could not update local conversation storage: {error}"))?;
    }
    Ok(())
}

fn ensure_conversation_management_columns(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(conversations)")
        .map_err(|e| e.to_string())?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    drop(statement);
    for name in ["pinned", "archived"] {
        if !columns.iter().any(|column| column == name) {
            connection
                .execute(
                    &format!(
                        "ALTER TABLE conversations ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    ),
                    [],
                )
                .map_err(|error| format!("Could not update conversation controls: {error}"))?;
        }
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn load_or_create_key(path: &Path, database_exists: bool) -> Result<Vec<u8>, String> {
    use std::{fs::OpenOptions, io::Write};

    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::Cryptography::{
            BCryptGenRandom, CryptProtectData, CryptUnprotectData, BCRYPT_USE_SYSTEM_PREFERRED_RNG,
            CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
        },
    };

    if path.exists() {
        let protected = fs::read(path)
            .map_err(|error| format!("Could not read the protected database key: {error}"))?;
        if protected.is_empty() || protected.len() > 16 * 1024 {
            return Err("The protected database key file has an invalid size.".into());
        }
        let input = CRYPT_INTEGER_BLOB {
            cbData: protected.len() as u32,
            pbData: protected.as_ptr() as *mut u8,
        };
        let mut output = CRYPT_INTEGER_BLOB::default();
        let success = unsafe {
            CryptUnprotectData(
                &input,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut output,
            )
        };
        if success == 0 {
            if !output.pbData.is_null() {
                unsafe { LocalFree(output.pbData.cast()) };
            }
            return Err(
                "Windows could not unlock this device's database key for the current user.".into(),
            );
        }
        let key = if output.pbData.is_null() || output.cbData != 32 {
            Err("The protected database key has an invalid length.".to_string())
        } else {
            Ok(unsafe {
                std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec()
            })
        };
        if !output.pbData.is_null() {
            unsafe { LocalFree(output.pbData.cast()) };
        }
        return key;
    }

    if database_exists {
        return Err("The database exists but its protected key is missing; refusing to overwrite or recreate it.".into());
    }

    let mut key = [0u8; 32];
    let random_status = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            key.as_mut_ptr(),
            key.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if random_status < 0 {
        return Err(format!(
            "Windows could not generate a database encryption key (status {random_status:#x})."
        ));
    }
    let input = CRYPT_INTEGER_BLOB {
        cbData: key.len() as u32,
        pbData: key.as_mut_ptr(),
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    let success = unsafe {
        CryptProtectData(
            &input,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    key.fill(0);
    if success == 0 {
        if !output.pbData.is_null() {
            unsafe { LocalFree(output.pbData.cast()) };
        }
        return Err("Windows could not protect the database key for the current user.".into());
    }
    if output.pbData.is_null() || output.cbData == 0 {
        return Err("Windows returned an empty protected database key.".into());
    }
    let protected = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) };
    let file_result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(protected)?;
        file.sync_all()
    })();
    unsafe { LocalFree(output.pbData.cast()) };
    file_result.map_err(|error| format!("Could not save the protected database key: {error}"))?;

    // Unwrap the newly protected key through the same user-scoped DPAPI path.
    let protected = fs::read(path).map_err(|error| error.to_string())?;
    let input = CRYPT_INTEGER_BLOB {
        cbData: protected.len() as u32,
        pbData: protected.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    let success = unsafe {
        CryptUnprotectData(
            &input,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if success == 0 || output.pbData.is_null() || output.cbData != 32 {
        if !output.pbData.is_null() {
            unsafe { LocalFree(output.pbData.cast()) };
        }
        return Err("Windows could not verify the newly protected database key.".into());
    }
    let key = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe { LocalFree(output.pbData.cast()) };
    Ok(key)
}

#[cfg(not(target_os = "windows"))]
fn load_or_create_key(_path: &Path, _database_exists: bool) -> Result<Vec<u8>, String> {
    Err("Encrypted local history is not yet supported on this platform.".into())
}
