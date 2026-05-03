use crate::config::{AppConfig, ModelConfig};
use std::{process::Command, process::Stdio};

pub fn start_llamafile(config: &AppConfig, model: &ModelConfig) -> std::process::Child {
    let llama = expand_path(&config.llamafile_path);
    let model_path = expand_path(&model.path);

    println!("llamafile_path = {:?}", llama);
    println!("model_path = {:?}", model_path);
    println!("exists = {}", std::fs::metadata(&llama).is_ok());
    println!("canonical = {:?}", std::fs::canonicalize(&llama).ok());
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

    // 轮询检查端口是否已经 Ready
    let url = format!(
        "http://{}:{}/health",
        config.server.host, config.server.port
    );
    println!("Waiting for LLM engine to initialize...");

    for _ in 0..30 {
        // 最多等 30 秒
        if ureq::get(&url).call().is_ok() {
            println!("Engine ready!");
            return child;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));

        // 顺便检查子进程是否中途崩溃了
        if let Ok(Some(status)) = child.try_wait() {
            panic!("Llamafile exited early with status: {}", status);
        }
    }
    child
}

fn expand_path(path: &str) -> String {
    shellexpand::tilde(path).to_string()
}
