use gpui::{Hsla, rgba};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use terminal::ColorPalette;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreference {
    #[default]
    System,
    Light,
    Dark,
}

impl ThemePreference {
    pub fn label(&self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Light => "Light",
            Self::Dark => "Dark",
        }
    }

    #[allow(dead_code)]
    pub fn description(&self) -> &'static str {
        match self {
            Self::System => "Automatically match system appearance",
            Self::Light => "Clean, crisp light theme",
            Self::Dark => "Sleek high-contrast dark theme",
        }
    }

    pub fn resolve_mode(&self, appearance: gpui::WindowAppearance) -> gpui_component::ThemeMode {
        match self {
            Self::System => appearance.into(),
            Self::Light => gpui_component::ThemeMode::Light,
            Self::Dark => gpui_component::ThemeMode::Dark,
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct ThemeTokens {
    pub is_dark: bool,

    // Window & background
    pub window_bg: Hsla,
    pub text_primary: Hsla,
    pub text_secondary: Hsla,
    pub text_muted: Hsla,
    pub border: Hsla,
    pub border_subtle: Hsla,

    // Titlebar
    pub titlebar_bg: Hsla,
    pub titlebar_border: Hsla,
    pub titlebar_fg: Hsla,
    pub titlebar_btn_hover: Hsla,

    // Sidebar
    pub sidebar_bg: Hsla,
    pub sidebar_bg_start: Hsla,
    pub sidebar_bg_end: Hsla,
    pub sidebar_border: Hsla,
    pub sidebar_fg: Hsla,
    pub sidebar_muted: Hsla,
    pub sidebar_item_hover: Hsla,
    pub sidebar_item_active: Hsla,
    pub sidebar_item_active_fg: Hsla,
    pub sidebar_section_header: Hsla,
    pub sidebar_icon_btn_hover: Hsla,
    pub sidebar_icon: Hsla,
    pub sidebar_icon_hover: Hsla,
    pub sidebar_footer_bg: Hsla,
    pub sidebar_footer_border: Hsla,

    // Terminal Panes
    pub pane_bg: Hsla,
    pub pane_border: Hsla,
    pub pane_border_focused: Hsla,
    pub pane_header_bg: Hsla,
    pub pane_header_fg: Hsla,
    pub pane_header_btn_hover: Hsla,
    pub pane_header_icon: Hsla,
    pub pane_header_icon_hover: Hsla,

    // Controls & Buttons
    pub primary_btn_bg: Hsla,
    pub primary_btn_hover: Hsla,
    pub primary_btn_fg: Hsla,
    pub secondary_btn_bg: Hsla,
    pub secondary_btn_hover: Hsla,
    pub secondary_btn_fg: Hsla,

    // Cards & Panels (Settings page & Dialogs)
    pub card_bg: Hsla,
    pub card_border: Hsla,
    pub card_hover: Hsla,
    pub pill_active_bg: Hsla,
    pub pill_active_fg: Hsla,
    pub pill_inactive_bg: Hsla,
    pub pill_inactive_fg: Hsla,
    pub ring: Hsla,
}

impl ThemeTokens {
    pub fn from_mode(mode: gpui_component::ThemeMode) -> Self {
        if mode.is_dark() {
            Self::dark()
        } else {
            Self::light()
        }
    }

    pub fn dark() -> Self {
        Self {
            is_dark: true,
            window_bg: rgba(0x14151bff).into(),
            text_primary: rgba(0xe6edf3ff).into(),
            text_secondary: rgba(0x8788a5ff).into(),
            text_muted: rgba(0x625b69ff).into(),
            border: rgba(0x22232cff).into(),
            border_subtle: rgba(0x20212aff).into(),

            titlebar_bg: rgba(0x14151bff).into(),
            titlebar_border: rgba(0x20212aff).into(),
            titlebar_fg: rgba(0x8788a5ff).into(),
            titlebar_btn_hover: rgba(0x23242cff).into(),

            sidebar_bg: rgba(0x121319ff).into(),
            sidebar_bg_start: rgba(0x121319ff).into(),
            sidebar_bg_end: rgba(0x121319ff).into(),
            sidebar_border: rgba(0x242531ff).into(),
            sidebar_fg: rgba(0xe6edf3ff).into(),
            sidebar_muted: rgba(0x8b879eff).into(),
            sidebar_item_hover: rgba(0xffffff15).into(),
            sidebar_item_active: rgba(0xffffff22).into(),
            sidebar_item_active_fg: rgba(0xffffffff).into(),
            sidebar_section_header: rgba(0x9a95adff).into(),
            sidebar_icon_btn_hover: rgba(0xffffff1a).into(),
            sidebar_icon: rgba(0x9a95adff).into(),
            sidebar_icon_hover: rgba(0xffffffff).into(),
            sidebar_footer_bg: rgba(0x13141aff).into(),
            sidebar_footer_border: rgba(0x242531ff).into(),

            pane_bg: rgba(0x0c0d12ff).into(),
            pane_border: rgba(0x22232cff).into(),
            pane_border_focused: rgba(0x38394eff).into(),
            pane_header_bg: rgba(0x0c0d12ff).into(),
            pane_header_fg: rgba(0x918995ff).into(),
            pane_header_btn_hover: rgba(0x23242cff).into(),
            pane_header_icon: rgba(0x817987ff).into(),
            pane_header_icon_hover: rgba(0xf0f6fcff).into(),

            primary_btn_bg: rgba(0x1f6febff).into(),
            primary_btn_hover: rgba(0x2f81f7ff).into(),
            primary_btn_fg: rgba(0xffffffff).into(),
            secondary_btn_bg: rgba(0x26272fff).into(),
            secondary_btn_hover: rgba(0x2d2e38ff).into(),
            secondary_btn_fg: rgba(0xffffffff).into(),

            card_bg: rgba(0x1a1b24ff).into(),
            card_border: rgba(0x2b2c3cff).into(),
            card_hover: rgba(0x222330ff).into(),
            pill_active_bg: rgba(0x1f6febff).into(),
            pill_active_fg: rgba(0xffffffff).into(),
            pill_inactive_bg: rgba(0x23242cff).into(),
            pill_inactive_fg: rgba(0x918995ff).into(),
            ring: rgba(0x388bfdff).into(),
        }
    }

    pub fn light() -> Self {
        Self {
            is_dark: false,
            window_bg: rgba(0xf6f8faff).into(),
            text_primary: rgba(0x1f2328ff).into(),
            text_secondary: rgba(0x59636eff).into(),
            text_muted: rgba(0x746d7aff).into(),
            border: rgba(0xd1d9e0ff).into(),
            border_subtle: rgba(0xe1e4e8ff).into(),

            titlebar_bg: rgba(0xffffffff).into(),
            titlebar_border: rgba(0xd1d9e0ff).into(),
            titlebar_fg: rgba(0x1f2328ff).into(),
            titlebar_btn_hover: rgba(0xeaeef2ff).into(),

            sidebar_bg: rgba(0xf6f8faff).into(),
            sidebar_bg_start: rgba(0xf6f8faff).into(),
            sidebar_bg_end: rgba(0xf6f8faff).into(),
            sidebar_border: rgba(0xd1d9e0ff).into(),
            sidebar_fg: rgba(0x1f2328ff).into(),
            sidebar_muted: rgba(0x656d76ff).into(),
            sidebar_item_hover: rgba(0x0000000a).into(),
            sidebar_item_active: rgba(0x00000014).into(),
            sidebar_item_active_fg: rgba(0x1f2328ff).into(),
            sidebar_section_header: rgba(0x656d76ff).into(),
            sidebar_icon_btn_hover: rgba(0x0000000e).into(),
            sidebar_icon: rgba(0x656d76ff).into(),
            sidebar_icon_hover: rgba(0x1f2328ff).into(),
            sidebar_footer_bg: rgba(0xf0f2f5ff).into(),
            sidebar_footer_border: rgba(0xd1d9e0ff).into(),

            pane_bg: rgba(0xffffffff).into(),
            pane_border: rgba(0xd1d9e0ff).into(),
            pane_border_focused: rgba(0x0969daff).into(),
            pane_header_bg: rgba(0xf6f8faff).into(),
            pane_header_fg: rgba(0x59636eff).into(),
            pane_header_btn_hover: rgba(0xeaeef2ff).into(),
            pane_header_icon: rgba(0x59636eff).into(),
            pane_header_icon_hover: rgba(0x1f2328ff).into(),

            primary_btn_bg: rgba(0x0969daff).into(),
            primary_btn_hover: rgba(0x0854b0ff).into(),
            primary_btn_fg: rgba(0xffffffff).into(),
            secondary_btn_bg: rgba(0xeaeef2ff).into(),
            secondary_btn_hover: rgba(0xdfe4e8ff).into(),
            secondary_btn_fg: rgba(0x1f2328ff).into(),

            card_bg: rgba(0xffffffff).into(),
            card_border: rgba(0xd1d9e0ff).into(),
            card_hover: rgba(0xf6f8faff).into(),
            pill_active_bg: rgba(0x0969daff).into(),
            pill_active_fg: rgba(0xffffffff).into(),
            pill_inactive_bg: rgba(0xeaeef2ff).into(),
            pill_inactive_fg: rgba(0x59636eff).into(),
            ring: rgba(0x0969daff).into(),
        }
    }
}

pub fn parse_hex(s: &str) -> Option<u32> {
    let hex = s.strip_prefix('#')?;
    if hex.len() == 6 {
        u32::from_str_radix(&format!("{hex}ff"), 16).ok()
    } else if hex.len() == 8 {
        u32::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

pub fn hex_to_rgb(hex: u32) -> (u8, u8, u8) {
    (
        ((hex >> 24) & 0xff) as u8,
        ((hex >> 16) & 0xff) as u8,
        ((hex >> 8) & 0xff) as u8,
    )
}

pub fn load_github_palette(dark: bool) -> ColorPalette {
    let theme_data = crate::assets::AppAssets::get("themes/github.json");
    let target_name = if dark { "GitHub Dark" } else { "GitHub Light" };
    if let Some(file) = theme_data
        && let Ok(Value::Array(themes)) = serde_json::from_slice::<Value>(&file.data)
        && let Some(target_theme) = themes
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some(target_name))
    {
                let mut builder = ColorPalette::builder();
                let style = target_theme.get("style").and_then(Value::as_object);

                let get_color = |key: &str| -> Option<(u8, u8, u8)> {
                    style
                        .and_then(|s| s.get(key))
                        .and_then(Value::as_str)
                        .and_then(parse_hex)
                        .map(hex_to_rgb)
                };

                if let Some((r, g, b)) = get_color("terminal.background")
                    .or_else(|| get_color("editor.background"))
                    .or_else(|| get_color("background"))
                {
                    builder = builder.background(r, g, b);
                }

                if let Some((r, g, b)) = get_color("terminal.foreground")
                    .or_else(|| get_color("editor.foreground"))
                    .or_else(|| get_color("text"))
                {
                    builder = builder.foreground(r, g, b);
                }

                let cursor_color = target_theme
                    .get("players")
                    .and_then(Value::as_array)
                    .and_then(|p| p.first())
                    .and_then(|p| p.get("cursor"))
                    .and_then(Value::as_str)
                    .and_then(parse_hex)
                    .map(hex_to_rgb)
                    .or_else(|| get_color("text.accent"))
                    .or_else(|| get_color("terminal.bright_foreground"))
                    .or_else(|| get_color("terminal.foreground"));

                if let Some((r, g, b)) = cursor_color {
                    builder = builder.cursor(r, g, b);
                }

                let ansi_keys = [
                    ("terminal.ansi.black", 0),
                    ("terminal.ansi.red", 1),
                    ("terminal.ansi.green", 2),
                    ("terminal.ansi.yellow", 3),
                    ("terminal.ansi.blue", 4),
                    ("terminal.ansi.magenta", 5),
                    ("terminal.ansi.cyan", 6),
                    ("terminal.ansi.white", 7),
                    ("terminal.ansi.bright_black", 8),
                    ("terminal.ansi.bright_red", 9),
                    ("terminal.ansi.bright_green", 10),
                    ("terminal.ansi.bright_yellow", 11),
                    ("terminal.ansi.bright_blue", 12),
                    ("terminal.ansi.bright_magenta", 13),
                    ("terminal.ansi.bright_cyan", 14),
                    ("terminal.ansi.bright_white", 15),
                ];

                for (key, idx) in ansi_keys {
                    if let Some((r, g, b)) = get_color(key) {
                        builder = match idx {
                            0 => builder.black(r, g, b),
                            1 => builder.red(r, g, b),
                            2 => builder.green(r, g, b),
                            3 => builder.yellow(r, g, b),
                            4 => builder.blue(r, g, b),
                            5 => builder.magenta(r, g, b),
                            6 => builder.cyan(r, g, b),
                            7 => builder.white(r, g, b),
                            8 => builder.bright_black(r, g, b),
                            9 => builder.bright_red(r, g, b),
                            10 => builder.bright_green(r, g, b),
                            11 => builder.bright_yellow(r, g, b),
                            12 => builder.bright_blue(r, g, b),
                            13 => builder.bright_magenta(r, g, b),
                            14 => builder.bright_cyan(r, g, b),
                            15 => builder.bright_white(r, g, b),
                            _ => builder,
                        };
                    }
                }

                return builder.build();
    }

    if dark {
        ColorPalette::builder()
            .background(0x01, 0x04, 0x09)
            .foreground(0xf0, 0xf6, 0xfc)
            .cursor(0x44, 0x93, 0xf8)
            .build()
    } else {
        ColorPalette::builder()
            .background(0xff, 0xff, 0xff)
            .foreground(0x1f, 0x23, 0x28)
            .cursor(0x09, 0x69, 0xda)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_preference_default_is_system() {
        assert_eq!(ThemePreference::default(), ThemePreference::System);
    }

    #[test]
    fn test_theme_tokens_modes() {
        let dark = ThemeTokens::dark();
        assert!(dark.is_dark);
        assert!(dark.sidebar_bg.l < 0.2);

        let light = ThemeTokens::light();
        assert!(!light.is_dark);
        assert!(light.sidebar_bg.l > 0.8, "Light sidebar should be directly light like other places");
        assert_eq!(light.sidebar_bg, light.window_bg);
    }

    #[test]
    fn test_load_github_palette_dark_and_light() {
        let dark_palette = load_github_palette(true);
        let light_palette = load_github_palette(false);
        // Dark background should be very dark, light background very light
        assert!(dark_palette.background().l < 0.2);
        assert!(light_palette.background().l > 0.8);
    }
}
