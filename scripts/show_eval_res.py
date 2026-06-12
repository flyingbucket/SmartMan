from pathlib import Path
import pandas as pd


def merge_and_deduplicate(res_dir="./eval_results", output_name="merged_report.csv"):
    eval_res_dir = Path(res_dir)

    # 1. 找出目录下所有的 csv 文件
    csv_files = list(eval_res_dir.glob("*.csv"))
    if not csv_files:
        print("未找到任何 CSV 文件！")
        return None

    print(f"找到 {len(csv_files)} 个 CSV 文件，准备读取...")

    # 2. 读取并纵向拼接所有 DataFrame
    df_list = []
    for file in csv_files:
        try:
            # 兼容空文件或损坏的文件
            df = pd.read_csv(file)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"读取文件 {file.name} 失败，跳过。错误: {e}")

    if not df_list:
        print("没有有效的 DataFrame 可以合并。")
        return None

    all_df = pd.concat(df_list, ignore_index=True)
    total_before = len(all_df)

    # 3. 关键步骤：按照你指定的 4 个 Key 规则进行去重
    # 先填充 NaN，防止 inplace 因为空值导致误判（如果是 string 建议统一转为 str 处理）
    dup_keys = ["base_model", "quant_config", "lora_path", "data_dirs"]
    for key in dup_keys:
        if key in all_df.columns:
            all_df[key] = all_df[key].fillna("").astype(str)

    # 如果有 timestamp 列，先排序，保证去重时保留的是最新那次跑出来的结果
    if "timestamp" in all_df.columns:
        all_df = all_df.sort_values(by="timestamp", ascending=True)

    # drop_duplicates 会根据 subset 去重，keep='last' 表示保留最新时间戳的一条
    deduped_df = all_df.drop_duplicates(subset=dup_keys, keep="last")

    total_after = len(deduped_df)
    print(
        f"合并完成！去重前总行数: {total_before} -> 去重后总行数: {total_after} (删除了 {total_before - total_after} 条重复数据)"
    )

    # 4. 写回新的总表或覆盖原始历史
    output_path = eval_res_dir / output_name
    deduped_df.to_csv(output_path, index=False)
    print(f"结果已成功保存至: {output_path}")

    return deduped_df


if __name__ == "__main__":
    # 执行合并
    final_df = merge_and_deduplicate()
