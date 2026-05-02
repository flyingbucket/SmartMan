use arboard::Clipboard;
use clap::Parser;
use colored::*;
use inquire::Text;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
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
