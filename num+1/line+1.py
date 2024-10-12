import re
import pyperclip


def increment_string(s):
    numbers = re.findall(r"\d+", s)
    for number in numbers:
        new_number = str(int(number) + 1)
        s = s.replace(number, new_number, 1)
    return s


while True:
    user_input = input("请输入一串字符（输入'结束'以停止程序）: ")
    if user_input.lower() == "结束":
        print("程序已结束")
        break
    result = increment_string(user_input)
    pyperclip.copy(result)  # 将结果复制到剪贴板
    print("处理后的字符串已复制到剪贴板:", result)
