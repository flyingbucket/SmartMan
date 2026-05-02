use clap::Parser;
use colored::*;
use inquire::{Select, Text};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(author, version, about = "SmartMan: NL2Bash CLI Assistant")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "http://127.0.0.1:8080/v1/chat/completions"
    )]
    api_url: String,
}

// 对应 OpenAI API 格式的请求结构
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    stream: bool, // 我们先用非流式，逻辑更简单
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

fn main() {
    let args = Args::parse();
    let re = Regex::new(r"```(?:bash)?\n?([\s\S]*?)```").unwrap();

    println!("{}", "── SmartMan CLI v0.1.0 ──".bold().blue());
    println!("{}", "提示: 输入 'exit' 或 'quit' 退出程序".dimmed());

    loop {
        // 1. 获取用户输入
        let prompt = match Text::new(" 如何帮您？").prompt() {
            Ok(val) => val,
            Err(_) => break, // 处理 Ctrl-C
        };

        if prompt == "exit" || prompt == "quit" {
            break;
        }
        if prompt.trim().is_empty() {
            continue;
        }

        // 2. 构造请求
        let req_body = ChatRequest {
            model: "smartman".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            temperature: 0.0, // 强制 0 温度，保证 Bash 输出稳定
            stream: false,
        };

        println!("{}", "󱙺 思考中...".dimmed());

        // 3. 发送请求 (使用 ureq 3.x 语法)
        let response = ureq::post(&args.api_url).send_json(&req_body);

        match response {
            Ok(res) => {
                let chat_res: ChatResponse = res.into_json().unwrap();
                let content = &chat_res.choices[0].message.content;

                // 4. 提取并展示命令
                let cmd = if let Some(caps) = re.captures(content) {
                    caps.get(1).map_or(content.trim(), |m| m.as_str().trim())
                } else {
                    content.trim()
                };

                println!(
                    "\n{}\n  {}\n",
                    "  推荐命令:".yellow().bold(),
                    cmd.cyan().italic()
                );

                // 5. 交互选项
                let options = vec!["执行命令", "复制命令", "放弃"];
                let ans = Select::new("选择操作:", options).prompt();

                match ans {
                    Ok("执行命令") => {
                        println!("{}", "正在执行...".dimmed());
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
                            Err(e) => eprintln!("执行失败: {}", e),
                        }
                    }
                    Ok("复制命令") => {
                        // 提示：若需实现剪贴板功能可添加 arboard crate
                        println!("{}", "功能开发中，请手动复制以上内容。".dimmed());
                    }
                    _ => println!("{}", "已取消。".dimmed()),
                }
            }
            Err(e) => {
                eprintln!("\n{} {}", "󰚼 API 连接失败:".red().bold(), e);
                eprintln!("请确保 llamafile --server 已在 {} 运行\n", args.api_url);
            }
        }
    }
}
