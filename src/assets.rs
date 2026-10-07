use gpui::{px, App};
use rust_embed::RustEmbed;

pub const SANS_FONT: &str = "IBM Plex Sans";
pub const MONO_FONT: &str = "Lilex";

#[derive(RustEmbed)]
#[folder = "assets/"]
// Unreferenced by the app (verified: no `file_icons/`, `logo/`, `ezicode`,
// or `olova` path is ever loaded); excluding them keeps ~3 MB out of the
// binary and out of every asset enumeration.
#[exclude = "file_icons/*"]
#[exclude = "logo/*"]
pub struct AppAssets;

pub struct CombinedAssets;

impl gpui::AssetSource for CombinedAssets {
    fn load(&self, path: &str) -> gpui::Result<Option<std::borrow::Cow<'static, [u8]>>> {
        let clean = path
            .trim_start_matches("icons/")
            .trim_start_matches("assets/")
            .trim_start_matches('/');
        if let Some(file) = AppAssets::get(clean) {
            return Ok(Some(file.data));
        }
        if let Some(file) = AppAssets::get(path) {
            return Ok(Some(file.data));
        }
        // Debug-only fallback to the working tree, so newly placed files work
        // without a rebuild. Release builds go straight to the component
        // assets instead of probing the filesystem on every miss.
        #[cfg(debug_assertions)]
        {
            let disk_paths = [
                std::path::Path::new("assets").join(clean),
                std::path::Path::new(clean).to_path_buf(),
                std::path::Path::new(path).to_path_buf(),
            ];
            for disk_path in disk_paths {
                if let Ok(bytes) = std::fs::read(&disk_path) {
                    return Ok(Some(std::borrow::Cow::Owned(bytes)));
                }
            }
        }
        gpui_component_assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<gpui::SharedString>> {
        let mut list = Vec::new();
        for file in AppAssets::iter() {
            list.push(gpui::SharedString::from(file.to_string()));
        }
        if let Ok(other) = gpui_component_assets::Assets.list(path) {
            list.extend(other);
        }
        Ok(list)
    }
}

pub fn load_embedded_fonts(cx: &App) {
    let fonts: Vec<std::borrow::Cow<'static, [u8]>> = AppAssets::iter()
        .filter(|p| p.starts_with("fonts/") && p.ends_with(".ttf"))
        .filter_map(|p| AppAssets::get(&p).map(|f| f.data))
        .collect();
    let _ = cx.text_system().add_fonts(fonts);
}

pub fn sync_component_fonts(cx: &mut App) {
    let theme = gpui_component::Theme::global_mut(cx);
    theme.font_family = SANS_FONT.into();
    theme.mono_font_family = MONO_FONT.into();
    theme.mono_font_size = px(13.5);
}
