import re

# 文件路径变量
input_file_path = r"C:\Users\Excoldinwarm\Desktop\num+1\1.txt"  # 确保这里的路径是正确的


def increment_string(s):
    lines = s.split("\n")
    incremented_lines = []

    for line in lines:
        numbers = re.findall(r"\d+", line)
        for number in numbers:
            new_number = str(int(number) + 1)  # 假设我们每次加1
            line = line.replace(number, new_number)
        incremented_lines.append(line)

    return "\n".join(incremented_lines)


# 从文件读取文本
try:
    with open(input_file_path, "r", encoding="utf-8") as file:
        input_text = file.read()
except FileNotFoundError:
    print(f"未找到文件: {input_file_path}")

count = 2
while count <= 20:
    # 处理文本
    incremented_text = increment_string(input_text)
    input_text = incremented_text

    # 将结果写入到另一个文件
    output_file_path = (
        f"C:\\Users\\Excoldinwarm\\Desktop\\num+1\\{count}.txt"  # 输出文件的路径
    )
    with open(output_file_path, "w") as file:
        file.write(incremented_text)

    print(f"处理后的字符串已写入到 {output_file_path}")

    count += 1
