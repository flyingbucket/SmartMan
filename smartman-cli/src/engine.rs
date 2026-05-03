use colored::Colorize;

use crate::config::{AppConfig, ModelConfig};
use std::{process::Command, process::Stdio};

use std::fs;
use std::path::PathBuf;

pub enum EngineHandle {
    Existing {
        pid: u32,
    }, // 只是重连别人的
    Owned {
        pid: u32,
        child: std::process::Child,
    }, // 本次启动的
}

pub fn get_pid_file() -> PathBuf {
    let mut path = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push("smartman");
    let _ = fs::create_dir_all(&path);
    path.push("engine.pid");
    path
}

// 检查 PID 是否有效且确实是 llamafile
fn is_process_running(pid: u32) -> bool {
    let cmdline_path = format!("/proc/{}/cmdline", pid);
    if let Ok(cmdline) = fs::read_to_string(cmdline_path) {
        // 验证命令行中是否包含 llamafile 关键字
        return cmdline.contains("llamafile");
    }
    false
}

pub fn start_llamafile(config: &AppConfig, model: &ModelConfig) -> EngineHandle {
    let url = format!(
        "http://{}:{}/health",
        config.server.host, config.server.port
    );
    let pid_file = get_pid_file();

    if let Ok(content) = fs::read_to_string(&pid_file)
        && let Ok(pid) = content.parse::<u32>()
        && is_process_running(pid)
    {
        println!("{}", "✔ Connected to existing engine.".green());
        return EngineHandle::Existing { pid };
    }

    let llama = expand_path(&config.llamafile_path);
    let model_path = expand_path(&model.path);

    let mut child = Command::new("sh")
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
        .expect("Failed to start llamafile");

    let pid = child.id();
    let _ = fs::write(&pid_file, pid.to_string());

    println!("Waiting for LLM engine to initialize...");
    for _ in 0..30 {
        if ureq::get(&url).call().is_ok() {
            println!("Engine ready!");
            return EngineHandle::Owned { pid, child };
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
        if let Ok(Some(status)) = child.try_wait() {
            panic!("Llamafile exited early with status: {}", status);
        }
    }

    EngineHandle::Owned { pid, child }
}

fn expand_path(path: &str) -> String {
    shellexpand::tilde(path).to_string()
}

pub fn stop_engine_by_pid(pid: u32) {
    println!(
        "{}",
        format!("Sending termination signal to PID {}...", pid).dimmed()
    );
    // 使用 libc 或者直接调用 kill 命令
    let _ = std::process::Command::new("kill")
        .arg(pid.to_string())
        .status();

    // 清理 pid 文件
    let _ = fs::remove_file(get_pid_file());
}
// pub fn start_llamafile(config: &AppConfig, model: &ModelConfig) -> (Option<u32>, bool) {
//     let url = format!(
//         "http://{}:{}/health",
//         config.server.host, config.server.port
//     );
//     let pid_file = get_pid_file();
//
//     if let Ok(content) = fs::read_to_string(&pid_file)
//         && let Ok(pid) = content.parse::<u32>()
//         && is_process_running(pid)
//     {
//         println!("{}", "✔ Connected to existing engine.".green());
//         return (Some(pid), false); // 返回 PID，但不负责清理（由主进程决定）
//     }
//     // --- 2. 启动逻辑 ---
//     let llama = expand_path(&config.llamafile_path);
//     let model_path = expand_path(&model.path);
//
//     let mut child = Command::new("sh")
//         .arg(&llama)
//         .arg("-m")
//         .arg(model_path)
//         .arg("--server")
//         .arg("--host")
//         .arg(&config.server.host)
//         .arg("--port")
//         .arg(config.server.port.to_string())
//         .stdout(Stdio::null())
//         .stderr(Stdio::null())
//         .spawn()
//         .expect("Failed to start llamafile");
//
//     let pid = child.id();
//
//     // 写入 PID 文件
//     let _ = fs::write(&pid_file, pid.to_string());
//     println!("Waiting for LLM engine to initialize...");
//     for _ in 0..30 {
//         if ureq::get(&url).call().is_ok() {
//             println!("Engine ready!");
//             return (Some(pid), true); // 返回子进程句柄，后续负责清理
//         }
//         std::thread::sleep(std::time::Duration::from_secs(1));
//         if let Ok(Some(status)) = child.try_wait() {
//             panic!("Llamafile exited early with status: {}", status);
//         }
//     }
//
//     (Some(pid), true)
// }
