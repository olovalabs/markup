use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SavedProject {
    pub path: PathBuf,
    pub name: String,
    #[serde(default = "default_expanded")]
    pub expanded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedWorkspace {
    #[serde(default)]
    pub projects: Vec<SavedProject>,
    #[serde(default)]
    pub active_project: Option<PathBuf>,
    #[serde(default = "default_visible")]
    pub sidebar_visible: bool,
    #[serde(default = "default_width")]
    pub sidebar_width: f32,
}

impl Default for SavedWorkspace {
    fn default() -> Self {
        Self {
            projects: Vec::new(),
            active_project: None,
            sidebar_visible: default_visible(),
            sidebar_width: default_width(),
        }
    }
}

fn default_visible() -> bool {
    true
}
fn default_expanded() -> bool {
    true
}
fn default_width() -> f32 {
    330.0
}

pub fn state_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("com", "Markup", "Markup")
        .map(|dirs| dirs.data_dir().join("workspace.json"))
}

pub fn load(path: &Path) -> SavedWorkspace {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<SavedWorkspace>(&bytes).ok())
        .unwrap_or_default()
}

pub fn save_atomic(path: &Path, state: &SavedWorkspace) -> io::Result<()> {
    let _guard = WRITE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    save_locked(path, state)
}

pub fn save_if_current(
    path: &Path,
    state: &SavedWorkspace,
    generation: &AtomicU64,
    expected: u64,
) -> io::Result<bool> {
    let _guard = WRITE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if generation.load(Ordering::SeqCst) != expected {
        return Ok(false);
    }
    save_locked(path, state)?;
    Ok(true)
}

fn save_locked(path: &Path, state: &SavedWorkspace) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "state path has no parent"))?;
    fs::create_dir_all(parent)?;
    let bytes = serde_json::to_vec(state).map_err(io::Error::other)?;
    let temp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temp, bytes)?;
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(())
}
