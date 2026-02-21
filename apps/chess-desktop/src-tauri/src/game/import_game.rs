use std::{
    path::Path,
    fs::read_to_string
};
use tauri_plugin_dialog::{DialogExt, FilePath};
use tauri::{AppHandle, Runtime};


fn _import_game_file_dialog<R>(app: AppHandle<R>) -> Option<String>
where
    R: Runtime
{
    // Open the file selector dialog
    let file_path: Option<FilePath> = app
        .dialog()
        .file()
        .add_filter("Chess Game", &["json"])
        .blocking_pick_file();

    // Extract the path
    if let Some(path) = file_path {
        // Convert into a PathBuf
        let path_ref: &Path = path.as_path()?;

        // Import the game
        match read_to_string(path_ref) {
            Ok(g) => Some(g),
            Err(_) => None
        }
    }

    // User cancelled the dialog
    else { None }
}

#[tauri::command]
pub fn import_game<R>(app: AppHandle<R>, game_type: String) -> String
where
    R: Runtime
{
    // Get the game type
    match game_type.as_str() {
        "classic" => match _import_game_file_dialog::<R>(app) {
            Some(g) => serde_json::json!({
                "type": "json",
                "message": g
            }).to_string(),
            None => serde_json::json!({
                "type": "error",
                "message": "File not found"
            }).to_string()
        },
        _ => todo!()
    }
}