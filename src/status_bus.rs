//! Per-user, per-Markup-instance lifecycle event transport.
//!
//! The socket accepts small newline-delimited JSON events from integrations.
//! Each event is addressed to the pane generation that launched the agent; a
//! delayed event cannot affect a replacement pane or a second Markup window.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_EVENT_BYTES: usize = 16 * 1024;
const MAX_QUEUED_EVENTS: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusEvent {
    pub version: u8,
    pub app_id: String,
    pub pane_id: u64,
    pub pane_generation: String,
    pub reporter: String,
    pub reporter_generation: String,
    pub sequence: u64,
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub event: String,
    pub message: Option<String>,
    pub approval_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    pub version: u8,
    pub app_id: String,
    pub pane_id: u64,
    pub pane_generation: String,
    pub state: String,
    pub evidence: String,
    pub confidence: String,
    pub reporter: Option<String>,
    pub updated_ms: u128,
}

pub fn snapshot_path(bus: &BusIdentity, pane_id: usize, generation: &str) -> PathBuf {
    bus.socket
        .parent()
        .unwrap_or(Path::new("/tmp"))
        .join(format!(
            "snapshot-{}-{pane_id}-{generation}.json",
            bus.app_id
        ))
}

static EVENTS: LazyLock<Arc<Mutex<VecDeque<StatusEvent>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(VecDeque::new())));

#[derive(Debug, Clone)]
pub struct BusIdentity {
    pub socket: PathBuf,
    pub app_id: String,
}

pub fn new_generation() -> String {
    random_token()
}

pub fn start() -> std::io::Result<BusIdentity> {
    let uid = unsafe { libc::getuid() };
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("markup-{uid}"));
    std::fs::create_dir_all(&base)?;
    #[cfg(unix)]
    std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o700))?;
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("status-") && name.ends_with(".sock"))
                && UnixStream::connect(&path).is_err()
            {
                let _ = std::fs::remove_file(path);
            }
        }
    }
    let app_id = format!("{}-{}", uid, random_token());
    let socket = base.join(format!("status-{}.sock", std::process::id()));
    if socket.exists() {
        std::fs::remove_file(&socket)?;
    }
    let listener = UnixListener::bind(&socket)?;
    #[cfg(unix)]
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    let queue = Arc::clone(&EVENTS);
    std::thread::Builder::new()
        .name("markup-status-socket".into())
        .spawn(move || {
            for connection in listener.incoming() {
                let Ok(stream) = connection else { continue };
                if !peer_is_same_user(&stream) {
                    continue;
                }
                let queue = Arc::clone(&queue);
                let _ = std::thread::Builder::new()
                    .name("markup-status-event".into())
                    .spawn(move || {
                        read_events(stream, queue);
                    });
            }
        })?;
    Ok(BusIdentity { socket, app_id })
}

fn peer_is_same_user(stream: &UnixStream) -> bool {
    #[cfg(target_os = "linux")]
    {
        let mut credentials: libc::ucred = unsafe { std::mem::zeroed() };
        let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        let result = unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut credentials as *mut libc::ucred).cast(),
                &mut length,
            )
        };
        return result == 0 && credentials.uid == unsafe { libc::getuid() };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = stream;
        true
    }
}

use std::os::fd::AsRawFd;

fn read_events(stream: UnixStream, queue: Arc<Mutex<VecDeque<StatusEvent>>>) {
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = Vec::with_capacity(512);
        match reader
            .by_ref()
            .take(MAX_EVENT_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)
        {
            Ok(0) | Err(_) => return,
            Ok(n) if n <= MAX_EVENT_BYTES => {
                if let Ok(request) = serde_json::from_slice::<serde_json::Value>(&line)
                    && request.get("method").is_some()
                {
                    let (events, response) = crate::herdr_compat::request(request);
                    {
                        let mut queue = queue.lock();
                        for event in events.into_iter().filter(valid) {
                            if queue.len() >= MAX_QUEUED_EVENTS {
                                queue.pop_front();
                            }
                            queue.push_back(event);
                        }
                    }
                    let _ = serde_json::to_writer(reader.get_mut(), &response);
                    let _ = reader.get_mut().write_all(b"\n");
                    continue;
                }
                if let Ok(event) = serde_json::from_slice::<StatusEvent>(&line)
                    && valid(&event)
                {
                    let mut queue = queue.lock();
                    if queue.len() >= MAX_QUEUED_EVENTS {
                        queue.pop_front();
                    }
                    queue.push_back(event);
                }
            }
            _ => return,
        }
    }
}

fn valid(event: &StatusEvent) -> bool {
    event.version == 1
        && event.pane_id > 0
        && event.sequence > 0
        && !event.app_id.is_empty()
        && event.app_id.len() <= 128
        && !event.pane_generation.is_empty()
        && event.pane_generation.len() <= 128
        && !event.reporter.is_empty()
        && event.reporter.len() <= 64
        && !event.reporter_generation.is_empty()
        && event.reporter_generation.len() <= 128
        && event
            .session_id
            .as_ref()
            .is_none_or(|id| id.len() <= 256 && !id.chars().any(char::is_control))
        && event
            .turn_id
            .as_ref()
            .is_none_or(|id| id.len() <= 256 && !id.chars().any(char::is_control))
        && matches!(
            event.event.as_str(),
            "working"
                | "idle"
                | "blocked"
                | "approval_resolved"
                | "done"
                | "error"
                | "clear"
                | "cancelled"
                | "register"
                | "heartbeat"
        )
}

pub fn drain() -> Vec<StatusEvent> {
    EVENTS.lock().drain(..).collect()
}

pub fn send(socket: &Path, event: &StatusEvent) -> std::io::Result<()> {
    let mut stream = UnixStream::connect(socket)?;
    serde_json::to_writer(&mut stream, event)?;
    stream.write_all(b"\n")?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn herdr_socket_requests_normalize_and_reply_before_the_next_report() {
        use serde_json::json;
        let token = "socket-compatibility-test";
        crate::herdr_compat::register("socket-test-app", 21, token);
        let (client, server) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let output = Arc::clone(&queue);
        let worker = std::thread::spawn(move || read_events(server, output));
        let mut client = BufReader::new(client);
        for (sequence, state, expected) in [
            (1, "working", "working"),
            (2, "blocked", "blocked"),
            (3, "working", "working"),
            (4, "idle", "done"),
        ] {
            let request = json!({"id":sequence.to_string(),"method":"pane.report_agent","params":{
                "pane_id":token,"source":"custom:cmd","agent":"cmd","seq":sequence,
                "agent_session_id":"native-session","state":state
            }});
            serde_json::to_writer(client.get_mut(), &request).unwrap();
            client.get_mut().write_all(b"\n").unwrap();
            let mut reply = String::new();
            client.read_line(&mut reply).unwrap();
            let reply: serde_json::Value = serde_json::from_str(&reply).unwrap();
            assert_eq!(reply["ok"], true);
            let mut events = queue.lock();
            let latest = events.back().unwrap();
            assert_eq!(latest.event, expected);
            assert_eq!(latest.app_id, "socket-test-app");
            assert_eq!(latest.pane_id, 21);
            assert!(events.iter().all(valid));
            events.clear();
        }
        crate::herdr_compat::unregister(token);
        drop(client);
        worker.join().unwrap();
    }

    fn valid_event() -> StatusEvent {
        StatusEvent {
            version: 1,
            app_id: "app".into(),
            pane_id: 4,
            pane_generation: "pane-run".into(),
            reporter: "codex".into(),
            reporter_generation: "session-1".into(),
            sequence: 2,
            session_id: Some("session-1".into()),
            turn_id: Some("turn-2".into()),
            event: "working".into(),
            message: None,
            approval_id: None,
        }
    }

    #[test]
    fn validates_version_identity_and_lifecycle_event() {
        assert!(valid(&valid_event()));
        let mut event = valid_event();
        event.version = 2;
        assert!(!valid(&event));
        let mut event = valid_event();
        event.event = "finished-ish".into();
        assert!(!valid(&event));
        let mut event = valid_event();
        event.sequence = 0;
        assert!(!valid(&event));
        let mut event = valid_event();
        event.turn_id = Some("bad\nturn".into());
        assert!(!valid(&event));
    }
}

fn random_token() -> String {
    let mut bytes = [0u8; 12];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut bytes))
        .is_err()
    {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        bytes.copy_from_slice(&nanos.to_le_bytes()[..12]);
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
