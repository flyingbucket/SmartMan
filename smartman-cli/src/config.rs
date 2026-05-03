use directories::ProjectDirs;
use serde::Deserialize;
use std::io;
#[cfg(unix)]
use std::os::unix::fs::symlink;
#[cfg(windows)]
use std::os::windows::fs::symlink_file as symlink;
use std::{env, fs, path::PathBuf}; // Windows 下区分文件和目录链接
const DEFAULT_CONFIG_YAML: &str = r#"
llamafile_path: "./assets/llamafile"
server:
  host: "127.0.0.1"
  port: 8080

model_dir: ./assets/models
"#;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub llamafile_path: String,
    pub server: ServerConfig,
    pub model_dir: String,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub path: String,
}

/// 将相对路径转换为相对于当前可执行文件的绝对路径
fn resolve_asset_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_absolute() {
        return path.to_string();
    }

    // 获取当前 exe 的绝对路径，例如 /usr/local/bin/smartman
    if let Ok(mut exe_path) = env::current_exe() {
        // 弹出文件名，得到 /usr/local/bin/
        exe_path.pop();
        // 拼接相对路径，得到 /usr/local/bin/assets/...
        return exe_path.join(path).to_string_lossy().to_string();
    }

    // 如果获取 exe 路径失败，回退到普通 shell 扩展
    shellexpand::tilde(path).to_string()
}
// 辅助函数：扫描目录下的 gguf 文件
fn discover_models(dir_path: &str) -> Vec<ModelConfig> {
    let mut models = Vec::new();

    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            // 仅匹配文件后缀为 .gguf 的项
            if path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("gguf")
                && let Some(file_name) = path.file_name().and_then(|s| s.to_str())
            {
                models.push(ModelConfig {
                    name: file_name.to_string(),
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }
    models
}

pub fn load_config() -> AppConfig {
    let config_path = config_path();
    let content = config_path
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .unwrap_or_else(|| DEFAULT_CONFIG_YAML.to_string());

    let mut config: AppConfig = serde_yaml::from_str(&content).expect("Invalid config yaml");
    config.llamafile_path = resolve_asset_path(&config.llamafile_path);
    config.model_dir = resolve_asset_path(&config.model_dir);

    config
}

fn config_path() -> Option<PathBuf> {
    ProjectDirs::from("com", "smartman", "smartman-cli")
        .map(|dirs| dirs.config_dir().join("config.yaml"))
}

pub fn select_model(config: &AppConfig) -> ModelConfig {
    let models = discover_models(&config.model_dir);

    if models.is_empty() {
        panic!(
            "No .gguf models found in {}.\nPlease install some models first.\nRun with -h for help",
            config.model_dir
        );
    }

    let options: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let ans = inquire::Select::new("Choose model:", options).prompt();

    match ans {
        Ok(name) => models.into_iter().find(|m| m.name == name).unwrap(),
        Err(_) => models[0].clone(),
    }
}

pub fn handle_install(target_type: &str, source: &str, use_link: bool) -> io::Result<()> {
    let exe_dir = std::env::current_exe()?
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Cannot find exe dir"))?
        .to_path_buf();

    // 1. 确定目标目录和固定的目标文件名
    let (target_dir, dest_name) = if target_type == "engine" {
        // 如果是引擎，固定命名为 "llamafile"
        (exe_dir.join("assets"), Some("llamafile".to_string()))
    } else {
        // 如果是模型，保持目录结构，文件名稍后动态获取
        (exe_dir.join("assets").join("models"), None)
    };

    fs::create_dir_all(&target_dir)?;

    let source_path = PathBuf::from(shellexpand::tilde(source).to_string());

    let file_name = source_path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid source filename"))?;

    // 如果 dest_name 有值（即引擎），用固定名；否则用原始文件名
    let final_dest_name = dest_name.unwrap_or_else(|| file_name.to_string_lossy().to_string());
    let dest_path = target_dir.join(final_dest_name);

    if use_link {
        #[cfg(unix)]
        {
            if dest_path.exists() {
                fs::remove_file(&dest_path)?;
            }
            // 注意：创建符号链接建议使用绝对路径以防失效
            let abs_source = fs::canonicalize(&source_path)?;
            symlink(&abs_source, &dest_path)?;
            println!("Created symlink: {:?} -> {:?}", abs_source, dest_path);
        }
        #[cfg(windows)]
        {
            if dest_path.exists() {
                fs::remove_file(&dest_path)?;
            }
            fs::copy(&source_path, &dest_path)?;
            println!(
                "Windows detected: Copied file instead of symlink: {:?}",
                dest_path
            );
        }
    } else {
        println!("Copying file (this may take a while)...");
        fs::copy(&source_path, &dest_path)?;
        println!("Installed to: {:?}", dest_path);
    }

    #[cfg(unix)]
    if target_type == "engine" {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&dest_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dest_path, perms)?;
    }

    Ok(())
}
