import re
import os

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            # 移除空行和标题行（以=开头的行）
            if line and not re.match(r'^\s*=+.*=+\s*$', line):
                # 基本清理（可选）
                line = re.sub(r'<[^>]+>', '', line)  # 移除HTML标签
                line = re.sub(r'\s+', ' ', line)     # 合并多个空格
                f_out.write(line + '\n')

input_dir = "/mnt/share/Datasets/wikitext/wikitext-103-v1"
output_dir = "/mnt/share/Datasets/wikitext/wikitext-103-v1/wikitext-103-cleaned"

os.makedirs(output_dir, exist_ok=True)

clean_file(f"{input_dir}/wiki.train.tokens", f"{output_dir}/train.txt")
clean_file(f"{input_dir}/wiki.valid.tokens", f"{output_dir}/valid.txt")  # Fairseq需要valid.txt
clean_file(f"{input_dir}/wiki.test.tokens", f"{output_dir}/test.txt")