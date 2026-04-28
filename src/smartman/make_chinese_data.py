import json
from pathlib import Path

Path("./data/processed/nl2bash").mkdir(parents=True, exist_ok=True)

system_msg = {
    "role": "system",
    "content": "你是一个Linux专家，请根据用户的中文描述，输出对应的Bash命令。只输出命令本身，不要任何解释。"
}

# ============================================================
# 中文数据集
# 重点覆盖 AI 容易出错的场景，共 8 大类
# ============================================================
chinese_data = [

    # --------------------------------------------------------
    # 类别1：引号与转义（最高频错误）
    # AI 经常忘记引号、用错单双引号、漏转义特殊字符
    # --------------------------------------------------------
    ("搜索包含美元符号$的行", r"grep '\$' file.txt"),
    ("在文件中搜索字面上的点号，而不是正则中的任意字符", r"grep '\.' file.txt"),
    ("搜索包含反斜杠的行", r"grep '\\' file.txt"),
    ("搜索包含星号*的行", r"grep '\*' file.txt"),
    ("搜索包含方括号[的行", r"grep '\[' file.txt"),
    ("用sed把文件中所有点号替换成下划线", r"sed 's/\./\_/g' file.txt"),
    ("用sed删除文件中所有空白行", r"sed '/^\s*$/d' file.txt"),
    ("用sed删除每行开头的空格", r"sed 's/^[[:space:]]*//' file.txt"),
    ("用sed删除每行末尾的空格", r"sed 's/[[:space:]]*$//' file.txt"),
    ("在文件名包含空格的文件里搜索关键词", "grep 'keyword' 'my file.txt'"),
    ("把含有空格的目录名作为参数传给ls", "ls 'my documents'"),
    ("用find查找文件名包含空格的文件", "find . -name '* *'"),

    # --------------------------------------------------------
    # 类别2：find 命令（AI 经常搞错逻辑组合和动作）
    # --------------------------------------------------------
    ("查找7天内修改过的文件", "find . -type f -mtime -7"),
    ("查找超过30天没有访问过的文件", "find . -type f -atime +30"),
    ("查找权限为777的文件", "find . -type f -perm 777"),
    ("查找权限不是644的文件", "find . -type f ! -perm 644"),
    ("查找空文件", "find . -type f -empty"),
    ("查找空目录", "find . -type d -empty"),
    ("查找所有者是root的文件", "find . -user root"),
    ("查找大小在1M到10M之间的文件", "find . -type f -size +1M -size -10M"),
    ("查找.py或.sh文件", "find . -type f \\( -name '*.py' -o -name '*.sh' \\)"),
    ("查找.py文件但排除venv目录", "find . -path './venv' -prune -o -name '*.py' -print"),
    ("查找.log文件并删除（不进入子目录）", "find . -maxdepth 1 -name '*.log' -delete"),
    ("查找所有软链接", "find . -type l"),
    ("查找文件名以数字开头的文件", "find . -name '[0-9]*'"),
    ("查找最近10分钟内创建的文件", "find . -type f -cmin -10"),
    ("统计当前目录下（不含子目录）的文件数量", "find . -maxdepth 1 -type f | wc -l"),

    # --------------------------------------------------------
    # 类别3：管道与重定向（AI 经常搞混 > >> 2>&1 顺序）
    # --------------------------------------------------------
    ("把命令的标准输出和错误输出都写入同一个文件", "command > output.log 2>&1"),
    ("把标准错误丢弃，只保留标准输出", "command 2>/dev/null"),
    ("把标准输出丢弃，只保留错误输出", "command 1>/dev/null"),
    ("把错误输出追加到已有的日志文件", "command 2>> error.log"),
    ("同时在终端显示输出并写入文件", "command | tee output.log"),
    ("把输出追加到文件同时显示在终端", "command | tee -a output.log"),
    ("把文件内容作为命令的标准输入", "command < input.txt"),
    ("统计命令输出的行数", "command | wc -l"),
    ("把上一个命令的退出码打印出来", "echo $?"),
    ("用xargs把find结果传给rm删除", "find . -name '*.tmp' | xargs rm -f"),
    ("用xargs处理含有空格的文件名", "find . -name '*.txt' -print0 | xargs -0 rm"),

    # --------------------------------------------------------
    # 类别4：awk 与 sed 进阶（AI 最容易写错语法）
    # --------------------------------------------------------
    ("用awk打印文件的第一列和第三列", "awk '{print $1, $3}' file.txt"),
    ("用awk计算第二列数字的总和", "awk '{sum += $2} END {print sum}' file.txt"),
    ("用awk打印行数大于5的行", "awk 'NR > 5' file.txt"),
    ("用awk打印第3行到第10行", "awk 'NR>=3 && NR<=10' file.txt"),
    ("用awk打印不重复的第一列", "awk '!seen[$1]++' file.txt"),
    ("用awk以冒号为分隔符打印第二列", "awk -F':' '{print $2}' file.txt"),
    ("用awk打印包含关键词error的行的第三列", "awk '/error/{print $3}' file.txt"),
    ("用awk统计文件行数（类似wc -l）", "awk 'END{print NR}' file.txt"),
    ("用sed只替换每行第一次出现的foo", "sed 's/foo/bar/' file.txt"),
    ("用sed替换第3行的内容", "sed '3s/.*/new content/' file.txt"),
    ("用sed打印第5行到第10行", "sed -n '5,10p' file.txt"),
    ("用sed删除注释行（以#开头）", "sed '/^#/d' file.txt"),
    ("用sed在每行末尾添加分号", "sed 's/$/;/' file.txt"),
    ("用sed在第3行后面插入一行新内容", "sed '3a\\new line content' file.txt"),

    # --------------------------------------------------------
    # 类别5：变量与条件判断（AI 常犯空格错误）
    # --------------------------------------------------------
    ("判断文件是否存在，存在则打印yes", "[ -f file.txt ] && echo 'yes'"),
    ("判断目录是否存在，不存在则创建", "[ ! -d mydir ] && mkdir mydir"),
    ("判断变量是否为空", "[ -z \"$var\" ] && echo 'empty'"),
    ("判断变量是否不为空", "[ -n \"$var\" ] && echo 'not empty'"),
    ("判断两个字符串是否相等", "[ \"$a\" = \"$b\" ] && echo 'equal'"),
    ("判断数字a是否大于b", "[ \"$a\" -gt \"$b\" ] && echo 'a is greater'"),
    ("遍历当前目录下所有py文件并打印文件名", "for f in *.py; do echo \"$f\"; done"),
    ("循环10次打印数字1到10", "for i in $(seq 1 10); do echo $i; done"),
    ("把命令输出赋值给变量", "result=$(ls -la)"),
    ("把当前时间赋值给变量", "now=$(date '+%Y-%m-%d %H:%M:%S')"),
    ("获取当前脚本所在目录的绝对路径", "dir=$(cd $(dirname $0) && pwd)"),

    # --------------------------------------------------------
    # 类别6：进程与后台任务（AI 常漏参数）
    # --------------------------------------------------------
    ("后台运行脚本并忽略挂断信号（关闭终端不停止）", "nohup python3 script.py &"),
    ("后台运行并把输出写入日志", "nohup python3 script.py > run.log 2>&1 &"),
    ("查看后台任务列表", "jobs -l"),
    ("把后台任务调回前台", "fg %1"),
    ("暂停当前前台进程并放到后台", "把前台进程按Ctrl+Z后执行 bg %1"),
    ("根据进程名查找进程ID", "pgrep -f python3"),
    ("杀掉特定PID的进程", "kill -9 12345"),
    ("杀掉所有名为gunicorn的进程", "pkill -f gunicorn"),
    ("等待某个进程结束再继续", "wait 12345"),
    ("查看进程的完整启动命令", "ps -p 12345 -o cmd"),
    ("查看占用8080端口的进程", "lsof -i :8080"),
    ("每隔2秒执行一次命令", "watch -n 2 'df -h'"),

    # --------------------------------------------------------
    # 类别7：字符串处理（AI 经常把参数顺序搞错）
    # --------------------------------------------------------
    ("截取变量的前5个字符", "echo ${var:0:5}"),
    ("截取变量第3个字符之后的内容", "echo ${var:2}"),
    ("获取变量的字符串长度", "echo ${#var}"),
    ("把变量中的foo替换成bar", "echo ${var/foo/bar}"),
    ("把变量中所有的foo替换成bar", "echo ${var//foo/bar}"),
    ("把变量转为大写", "echo ${var^^}"),
    ("把变量转为小写", "echo ${var,,}"),
    ("删除变量末尾的.txt后缀", "echo ${var%.txt}"),
    ("删除变量中最后一个斜杠之后的部分（取目录名）", "echo ${var%/*}"),
    ("删除变量中第一个斜杠之前的部分（取文件名）", "echo ${var##*/}"),
    ("给变量设置默认值（变量为空时用默认值）", "echo ${var:-default}"),

    # --------------------------------------------------------
    # 类别8：文件内容与权限（AI 常犯的细节错误）
    # --------------------------------------------------------
    ("递归修改目录下所有文件权限为644", "find . -type f -exec chmod 644 {} \\;"),
    ("递归修改目录下所有目录权限为755", "find . -type d -exec chmod 755 {} \\;"),
    ("只修改目录权限不修改文件权限", "find . -type d -exec chmod 755 {} +"),
    ("给文件添加可执行权限（不改变其他权限）", "chmod +x script.sh"),
    ("查看文件的八进制权限", "stat -c '%a %n' file.txt"),
    ("比较两个文件的不同", "diff file1.txt file2.txt"),
    ("比较两个目录的差异", "diff -r dir1/ dir2/"),
    ("按行合并两个文件（类似粘贴列）", "paste file1.txt file2.txt"),
    ("取两个文件的交集（共同行）", "comm -12 <(sort file1.txt) <(sort file2.txt)"),
    ("取只在file1中有、file2没有的行", "comm -23 <(sort file1.txt) <(sort file2.txt)"),
    ("用md5sum校验文件完整性", "md5sum file.txt"),
    ("批量校验目录下所有文件的md5", "find . -type f | xargs md5sum > checksums.md5"),
    ("分割大文件，每份100行", "split -l 100 bigfile.txt part_"),
    ("查看文件编码格式", "file -i file.txt"),
    ("把文件从GBK转换为UTF-8编码", "iconv -f GBK -t UTF-8 input.txt -o output.txt"),
]

# ============================
# 转换成训练格式
# ============================
dataset = []
for nl, cm in chinese_data:
    dataset.append({
        "messages": [
            system_msg,
            {"role": "user",      "content": nl},
            {"role": "assistant", "content": cm}
        ]
    })

# ============================
# 保存
# ============================
out_path = Path("./data/processed/nl2bash/train_chinese.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"共生成 {len(dataset)} 条中文数据")
print(f"已保存到 {out_path}")
print("\n分类统计：")
categories = [
    ("引号与转义", 12),
    ("find命令", 15),
    ("管道与重定向", 11),
    ("awk与sed进阶", 14),
    ("变量与条件判断", 11),
    ("进程与后台任务", 11),
    ("字符串处理", 11),
    ("文件内容与权限", 13),
]
for cat, count in categories:
    print(f"  {cat}: {count} 条")
