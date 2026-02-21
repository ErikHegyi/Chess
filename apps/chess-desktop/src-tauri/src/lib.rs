mod game;
use game::import_game;


#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![import_game])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
