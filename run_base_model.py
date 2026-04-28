from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== 加载模型 ==========
MODEL_PATH = "./Qwen2.5-Coder-1.5B"  # 改成你的实际路径

print("加载中，请稍候...")
#Tokenizer (分词器) 把输入的自然语言转换成计算机能听懂的数字序列
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16, #用半精度浮点数加载，显存占用减半，速度更快，精度影响很小
    device_map="auto", #把模型自动分配到可用的GPU上
    trust_remote_code=True
)
print(f"模型加载完成！运行在: {model.device}")

# ========== 推理函数 ==========
def generate_bash(chinese_input, max_new_tokens=128):
    """输入中文描述，输出Bash命令"""
    
    # 这是给基座模型的提示词格式
    prompt = f"""将以下中文需求转换为Bash命令，只输出命令本身，不要解释。

需求：{chinese_input}
命令："""
#三引号 """： 允许你在字符串里换行
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,      # 低温度让输出更确定
            do_sample=True,  #是否采样，True表示生成时会有随机性，False则是贪心算法（每一步都选概率最高的词）。这里设置为True可以让输出更丰富多样。
            pad_token_id=tokenizer.eos_token_id, #填充符
            eos_token_id=tokenizer.eos_token_id, #结束符
        )
    
    # 只取新生成的部分。outputs 实际上包含了 [你的问题 + AI 的回答]
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 只取第一行（Bash命令通常是一行）
    return result.strip().split("\n")[0]

# ========== 测试用例 ==========
test_cases = [
    "列出当前目录下所有的.py文件",
    "查看系统内存使用情况",
    "把file.txt的内容按行倒序输出",
    "递归删除当前目录下所有__pycache__文件夹",
    "统计当前目录下有多少个.sh文件",
]

print("\n" + "="*50)
print("【基座模型测试 - 中文转Bash】")
print("="*50)

for i, case in enumerate(test_cases, 1):
    result = generate_bash(case)
    print(f"\n[{i}] 输入: {case}")
    print(f"    输出: {result}")

print("\n" + "="*50)
print("测试完成！以上是未微调的基座模型输出，作为基线参考。")