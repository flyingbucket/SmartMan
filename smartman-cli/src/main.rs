use arboard::Clipboard;
use colored::*;
use inquire::Text;
use regex::Regex;
use smartman_cli::*;

fn main() {
    let re = Regex::new(r"```(?:bash)?\n?([\s\S]*?)```").unwrap();
    let mut clipboard = Clipboard::new().ok();

    let config = load_config();
    let model = select_model(&config);
    let _child = start_llamafile(&config, &model);

    let api_url = format!(
        "http://{}:{}/v1/chat/completions",
        config.server.host, config.server.port
    );

    println!("{}", "── SmartMan CLI REPL v0.1.0 ──".bold().blue());
    println!(
        "{}",
        "Note: enter 'exit' or 'quit' to exit this REPL".dimmed()
    );
    loop {
        let prompt_label = "User > ".green().bold().to_string();
        let prompt = match Text::new(&prompt_label).prompt() {
            Ok(val) => val,
            Err(_) => break,
        };

        if prompt == "exit" || prompt == "quit" {
            break;
        }
        if prompt.trim().is_empty() {
            continue;
        }

        let req_body = ChatRequest {
            model: "smartman".to_string(),
            messages: vec![
                ChatMessage{
                    role:"system".to_string(),
                    content:"You are a professional Bash expert. You should return bash commands accoring to my request without explation.".to_string()
                },
                ChatMessage { role: "user".to_string(), content: prompt }
            ],
            temperature: 0.0,
            stream: false,
        };

        println!("{}", "Thinking...".dimmed());

        let response = ureq::post(&api_url).send_json(&req_body);
        match response {
            Ok(res) => handle_ok_response(res, &re, clipboard.as_mut()),
            Err(e) => {
                eprintln!("\n{} {}", "API no resopose:".red().bold(), e);
                eprintln!(
                    "Please make sure llamafile --server is running at {}\n",
                    api_url
                );
            }
        }
    }
}
