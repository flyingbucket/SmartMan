use arboard::Clipboard;
use clap::Parser;
use colored::*;
use directories::ProjectDirs;
use inquire::Text;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::{fs, path::PathBuf, process::Command, process::Stdio};
use ureq::Body;
use ureq::http::Response;

#[derive(Parser, Debug)]
#[command(author, version, about = "SmartMan: NL2Bash CLI Assistant")]
pub struct Args {
    #[arg(
        short,
        long,
        default_value = "http://127.0.0.1:8080/v1/chat/completions"
    )]
    pub api_url: String,
}

#[derive(Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub stream: bool,
}

#[derive(Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
}

#[derive(Deserialize)]
pub struct ResponseMessage {
    pub content: String,
}

pub fn handle_ok_response(res: Response<Body>, re: &Regex, clipboard: Option<&mut Clipboard>) {
    let chat_res: ChatResponse = res.into_body().read_json().unwrap();
    let content = &chat_res.choices[0].message.content;

    let cmd = if let Some(caps) = re.captures(content) {
        caps.get(1).map_or(content.trim(), |m| m.as_str().trim())
    } else {
        content.trim()
    };

    println!(
        "\n{}\n  {}\n",
        "SmartMan: ".yellow().bold(),
        cmd.cyan().italic()
    );

    println!("\n[y] execute  [n] don't execute  [c] copy to clipboard");
    let action = Text::new("Your choice:").prompt();
    match action.as_deref() {
        Ok("y") | Ok("Y") => execute_command(cmd),
        Ok("c") | Ok("C") => copy_to_clipboard(cmd, clipboard),
        _ => println!("{}", "Keep chating".dimmed()),
    }
}

fn copy_to_clipboard(text: &str, clipboard: Option<&mut arboard::Clipboard>) {
    // Wayland wl-copy
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WAYLAND_DISPLAY").is_some()
            && let Ok(mut child) = Command::new("wl-copy").stdin(Stdio::piped()).spawn()
            && let Some(mut stdin) = child.stdin.take()
            && stdin.write_all(text.as_bytes()).is_ok()
        {
            drop(stdin);
            println!("{}", "Copied to clipboard!".green());
            return;
        }
    }
    // other platform or Wayland backup
    if let Some(cb) = clipboard {
        if let Err(e) = cb.set_text(text.to_string()) {
            eprintln!("Copy failed: {}", e);
        } else {
            println!("{}", "Copied to clipboard!".green());
        }
    } else {
        eprintln!("Clipboard unavailable.");
    }
}

fn execute_command(cmd: &str) {
    println!("{}", "Executing...".dimmed());
    let output = if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", cmd]).output()
    } else {
        Command::new("bash").arg("-c").arg(cmd).output()
    };

    match output {
        Ok(out) => {
            println!("{}", String::from_utf8_lossy(&out.stdout));
            if !out.stderr.is_empty() {
                eprintln!("{}", String::from_utf8_lossy(&out.stderr).red());
            }
        }
        Err(e) => eprintln!("Failed!: {}", e),
    }
}

const DEFAULT_CONFIG_YAML: &str = r#"
llamafile_path: "/home/flyingbucket/CODE/SmartMan/deps/llamafile"
server:
  host: "127.0.0.1"
  port: 8080

models:
  - name: "smartman-v1-f16"
    path: "/home/flyingbucket/CODE/SmartMan/dist/gguf/smartman-v1-f16.gguf"
  - name: "smartman-v1-q4_k_m"
    path: "/home/flyingbucket/CODE/SmartMan/dist/gguf/smartman-v1-q4_k_m.gguf"
"#;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub llamafile_path: String,
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
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

pub fn load_config() -> AppConfig {
    let config_path = config_path();
    let content = config_path
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .unwrap_or_else(|| DEFAULT_CONFIG_YAML.to_string());

    serde_yaml::from_str(&content).expect("Invalid config yaml")
}

fn config_path() -> Option<PathBuf> {
    ProjectDirs::from("com", "smartman", "smartman-cli")
        .map(|dirs| dirs.config_dir().join("config.yaml"))
}

pub fn select_model(config: &AppConfig) -> ModelConfig {
    let options: Vec<String> = config.models.iter().map(|m| m.name.clone()).collect();
    let ans = inquire::Select::new("Choose model:", options).prompt();

    match ans {
        Ok(name) => config
            .models
            .iter()
            .find(|m| m.name == name)
            .unwrap()
            .clone(),
        Err(_) => config.models[0].clone(),
    }
}

pub fn start_llamafile(config: &AppConfig, model: &ModelConfig) -> std::process::Child {
    let llama = expand_path(&config.llamafile_path);
    let model_path = expand_path(&model.path);

    println!("llamafile_path = {:?}", llama);
    println!("model_path = {:?}", model_path);
    println!("exists = {}", std::fs::metadata(&llama).is_ok());
    println!("canonical = {:?}", std::fs::canonicalize(&llama).ok());
    Command::new("sh")
        .arg(&llama)
        .arg("-m")
        .arg(model_path)
        .arg("--server")
        .arg("--host")
        .arg(&config.server.host)
        .arg("--port")
        .arg(config.server.port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start llamafile")
}

fn expand_path(path: &str) -> String {
    shellexpand::tilde(path).to_string()
}
