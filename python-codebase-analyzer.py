"""
======================================================
=          优化版Python代码库分析工具                  =
======================================================

专用于分析Python项目的强化工具，提供详细的代码质量报告：
- 项目结构和组织分析
- 代码质量与复杂度指标
- 模块依赖关系分析
- Python特有反模式检测
- 命名规范检查
- 质量评分系统
- 彩色终端输出
- 详细代码行数统计

作者: Claude
更新: 模块化重构，增强分析功能
======================================================
"""

# 版本信息
__version__ = "3.0.0"

import os
import ast
import fnmatch
import json
import logging
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from enum import Enum
import argparse



# =================== 日志配置模块 ===================
class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AnalysisOptions:
    """分析功能开关配置类
    
    每个开关控制对应功能的计算和输出
    关闭功能后，不仅不显示结果，也不进行相关计算
    """
    
    def __init__(self):
        # 功能开关（控制功能的计算和输出）
        self.calculate_detailed_lines = True      # 计算详细行数分解
        self.detect_antipatterns = True           # 检测反模式
        self.check_naming_conventions = True      # 检查命名规范
        self.analyze_imports = True               # 分析导入依赖
        self.calculate_complexity = True          # 计算循环复杂度
        self.analyze_docstrings = True            # 分析文档字符串
        self.calculate_quality_score = True       # 计算质量评分
        self.build_dependency_graph = True        # 构建依赖关系图
        self.show_project_structure = True        # 显示项目结构
        
        # 日志级别（只控制计算过程中的日志输出）
        self.log_level = LogLevel.INFO
        
    def set_log_level(self, level: LogLevel):
        """设置日志级别"""
        self.log_level = level
        
    def disable_all_features(self):
        """禁用所有功能（只保留基本统计）"""
        self.calculate_detailed_lines = False
        self.detect_antipatterns = False
        self.check_naming_conventions = False
        self.analyze_imports = False
        self.calculate_complexity = False
        self.analyze_docstrings = False
        self.calculate_quality_score = False
        self.build_dependency_graph = False
        
    def enable_basic_features_only(self):
        """只启用基本功能"""
        self.disable_all_features()
        self.show_project_structure = True


class AnalysisLogger:
    """分析日志管理器
    
    只负责日志输出，不控制功能开关
    """
    
    def __init__(self, log_level: LogLevel):
        self.log_level = log_level
        self.setup_logger()
        
    def setup_logger(self):
        """设置日志记录器"""
        # 创建日志记录器
        self.logger = logging.getLogger('CodeAnalyzer')
        self.logger.setLevel(getattr(logging, self.log_level.value))
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level.value))
        
        # 设置日志格式
        if self.log_level == LogLevel.DEBUG:
            formatter = logging.Formatter(
                f'{Colors.GRAY}[%(asctime)s] %(levelname)s: %(message)s{Colors.RESET}',
                datefmt='%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                f'{Colors.GRAY}%(levelname)s: %(message)s{Colors.RESET}'
            )
            
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def debug(self, message: str):
        """输出调试信息"""
        self.logger.debug(message)
        
    def info(self, message: str):
        """输出信息"""
        self.logger.info(message)
        
    def warning(self, message: str):
        """输出警告"""
        self.logger.warning(f"{Colors.YELLOW}{message}{Colors.RESET}")
        
    def error(self, message: str):
        """输出错误"""
        self.logger.error(f"{Colors.RED}{message}{Colors.RESET}")
        
    def file_progress(self, file_path: str, current: int, total: int):
        """显示文件处理进度（DEBUG级别显示）"""
        if self.log_level == LogLevel.DEBUG:
            percentage = (current / total) * 100 if total > 0 else 0
            self.debug(f"处理文件 [{current}/{total}] ({percentage:.1f}%): {file_path}")

# ================== 日志配置模块结束 ==================


# =================== 配置模块 ===================
CONFIG = {
    "target_path": r".",  # 默认为当前目录
    "output_format": "text",  # text/markdown/json
    "exclude_dirs": [
        "__pycache__",
        ".git",
        ".vscode",
        ".idea",
        "venv",
        "env",
        ".env",
        ".venv",
        "virtualenv",
        "node_modules",
        "logs",
        "dist",
        "build",
        "*.egg-info",
    ],
    "exclude_files": [
        "*.pyc",
        "*.pyo",
        "*.log",
        "*.tmp",
        "*.bak",
        "*.zip",
        "*.7z",
        "Thumbs.db",
        ".DS_Store",
        ".gitignore",
        "python-codebase-analyzer.py",
        "directory-analyzer.py",
        "text-content-analyzer.py",
    ],
    "include_patterns": ["*.py"],  # 默认只包括Python文件
    "show_hidden": False,
    "max_workers": 4,
    "max_file_size_mb": 10,  # 跳过大于此大小的文件
    "color_output": True,  # 彩色终端输出
    
    # 功能开关（控制功能的计算和输出）
    "calculate_detailed_lines": True,      # 计算详细行数分解
    "detect_antipatterns": True,           # 检测反模式
    "check_naming_conventions": True,      # 检查命名规范
    "analyze_imports": True,               # 分析导入依赖
    "calculate_complexity": True,          # 计算循环复杂度
    "analyze_docstrings": True,            # 分析文档字符串
    "calculate_quality_score": True,       # 计算质量评分
    "build_dependency_graph": True,        # 构建依赖关系图
    "show_project_structure": True,        # 显示项目结构
    
    # 日志级别（只控制计算过程中的日志输出）
    "log_level": "INFO",  # DEBUG/INFO/WARNING/ERROR
}
# ================== 配置模块结束 ==================


# =================== 终端颜色支持 ===================
class Colors:
    """终端ANSI颜色代码"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    @staticmethod
    def supports_color():
        """检测当前终端是否支持颜色"""
        # Windows 10以上的cmd和PowerShell支持颜色
        # Linux和macOS终端通常支持颜色
        platform = sys.platform
        if platform == "win32":
            return sys.getwindowsversion().major >= 10
        return True

    @staticmethod
    def disable():
        """禁用颜色输出"""
        for attr in dir(Colors):
            if (
                not attr.startswith("_")
                and attr != "disable"
                and attr != "supports_color"
            ):
                setattr(Colors, attr, "")


# 如果终端不支持颜色，则禁用
if not Colors.supports_color():
    Colors.disable()


@dataclass
class NamingStats:
    """Python命名规范统计"""

    good_names: int = 0
    bad_names: int = 0
    snake_case_vars: int = 0
    camel_case_vars: int = 0
    pascal_case_classes: int = 0
    non_pascal_classes: int = 0
    naming_issues: List[str] = field(default_factory=list)


@dataclass
class CodeQuality:
    """Python特定代码质量指标"""

    has_docstring: bool = False
    has_type_hints: bool = False
    has_tests: bool = False
    antipatterns_count: int = 0
    antipatterns: List[str] = field(default_factory=list)
    naming_stats: NamingStats = field(default_factory=NamingStats)


@dataclass
class FileStats:
    """
    Python文件统计信息
    
    按照模块化设计原则，统一管理文件各类统计数据
    遵循PEP8命名规范，使用snake_case命名
    """

    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    docstring_lines: int = 0
    cyclomatic_complexity: int = 0
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0

    import_count: int = 0
    imports: Dict[str, List[str]] = field(default_factory=dict)

    quality: CodeQuality = field(default_factory=CodeQuality)
    quality_score: float = 0.0
    
    # 新增：详细行数分析
    executable_lines: int = 0  # 可执行代码行
    logical_lines: int = 0     # 逻辑代码行
    physical_lines: int = 0    # 物理代码行
    
    def get_line_distribution(self) -> Dict[str, float]:
        """获取代码行分布百分比"""
        if self.total_lines == 0:
            return {"code": 0.0, "comment": 0.0, "docstring": 0.0, "blank": 0.0}
        
        return {
            "code": round((self.code_lines / self.total_lines) * 100, 1),
            "comment": round((self.comment_lines / self.total_lines) * 100, 1),
            "docstring": round((self.docstring_lines / self.total_lines) * 100, 1),
            "blank": round((self.blank_lines / self.total_lines) * 100, 1),
        }


@dataclass
class ModuleInfo:
    """Python模块信息"""

    name: str
    path: str
    is_package: bool = False
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    public_symbols: List[str] = field(default_factory=list)
    stats: Optional[FileStats] = None


class LineCountAnalyzer:
    """
    代码行数分析器
    
    按照模块化设计原则，专门负责分析代码的各类行数统计
    遵循PEP8命名规范，提供详细的行数分析功能
    """
    
    @staticmethod
    def analyze_lines(code_content: str) -> Dict[str, int]:
        """
        分析代码内容的各类行数
        
        Args:
            code_content: 代码内容字符串
            
        Returns:
            包含各类行数统计的字典
        """
        lines = code_content.split('\n')
        stats = {
            'total_lines': len(lines),
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'docstring_lines': 0,
            'executable_lines': 0,
            'logical_lines': 0,
            'physical_lines': len(lines)
        }
        
        in_docstring = False
        docstring_delimiter = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 空行
            if not stripped:
                stats['blank_lines'] += 1
                continue
            
            # 检查文档字符串
            if not in_docstring:
                # 检查是否开始文档字符串
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_delimiter = stripped[:3]
                    in_docstring = True
                    stats['docstring_lines'] += 1
                    # 检查是否在同一行结束
                    if stripped.count(docstring_delimiter) >= 2:
                        in_docstring = False
                        docstring_delimiter = None
                    continue
            else:
                # 文档字符串内部
                stats['docstring_lines'] += 1
                if docstring_delimiter and stripped.endswith(docstring_delimiter):
                    in_docstring = False
                    docstring_delimiter = None
                continue
            
            # 注释行
            if stripped.startswith('#'):
                stats['comment_lines'] += 1
                continue
            
            # 代码行
            stats['code_lines'] += 1
            
            # 可执行行（不包括定义和声明）
            if LineCountAnalyzer._is_executable_line(stripped):
                stats['executable_lines'] += 1
            
            # 逻辑行（可能包含多个逻辑单元）
            logical_count = LineCountAnalyzer._count_logical_lines(stripped)
            stats['logical_lines'] += logical_count
        
        return stats
    
    @staticmethod
    def _is_executable_line(line: str) -> bool:
        """判断是否为可执行代码行"""
        # 定义和声明关键字
        definition_keywords = ['def ', 'class ', 'import ', 'from ', 'global ', 'nonlocal ']
        
        for keyword in definition_keywords:
            if line.startswith(keyword):
                return False
        
        # 装饰器
        if line.startswith('@'):
            return False
            
        return True
    
    @staticmethod
    def _count_logical_lines(line: str) -> int:
        """计算逻辑代码行数"""
        # 简化版：以分号分隔的语句作为多个逻辑行
        # 排除字符串内的分号
        in_string = False
        string_char = None
        logical_count = 1  # 至少一行
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif char == ';':
                    logical_count += 1
            else:
                if char == string_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        return logical_count


class AntiPatternDetector:
    """检测Python代码中的常见反模式
    
    按照PEP8规范和模块化设计原则，提供精准的反模式检测
    """

    PATTERNS = {
        "bare_except": (r"except\s*:", "使用裸except语句"),
        "mutable_default": (
            r"def\s+\w+\s*\(.*=\s*\[\s*\].*\)|\(.*=\s*\{\s*\}.*\)",
            "使用可变默认参数",
        ),
        "global_statement": (r"\bglobal\b", "使用global语句"),
        "eval_usage": (r"\beval\(", "使用eval()函数"),
        "exec_usage": (r"\bexec\(", "使用exec()函数"),
        "wildcard_import": (r"from\s+[\w.]+\s+import\s+\*", "使用通配符导入"),
        "exit_call": (r"\bexit\(", "使用exit()而非sys.exit()"),
        "print_debugging": (r"^\s*print\(", "可能存在调试用print语句"),
        "hardcoded_path": (
            # 只检测真正的硬编码文件路径，排除URL和Django路由
            r"(?<!['\"])((?:[A-Za-z]:\\[\\\w\s.-]+)|(?:^\s*['\"]?/(?:usr|home|var|etc|opt|tmp)/[\w/.-]+['\"]?\s*$))",
            "硬编码文件路径"
        ),
        "nested_function": (
            r"def\s+\w+\s*\([^)]*\):\s*[^\n]*\n\s+def\s+",
            "嵌套函数定义",
        ),
        "too_many_arguments": (r"def\s+\w+\s*\([^)]{80,}\)", "函数参数过多"),
        "long_line": (r"^.{120,}$", "代码行过长(>120字符)"),
        "multiple_statements": (r";.*\w", "多个语句在同一行"),
    }

    @staticmethod
    def find_antipatterns(code: str) -> List[Tuple[str, str]]:
        """在代码中查找反模式

        返回包含（行，描述）的元组列表
        """
        antipatterns = []
        lines = code.splitlines()
        
        for name, (pattern, description) in AntiPatternDetector.PATTERNS.items():
            # 跳过特殊情况
            if name == "print_debugging":
                # 只检测单独的print语句，排除函数内部和文档字符串中的print
                if "logging" in code.lower() or "logger" in code.lower():
                    continue
                    
            if name == "hardcoded_path":
                # 特殊处理硬编码路径检测
                AntiPatternDetector._detect_hardcoded_paths(lines, antipatterns)
                continue
                
            if name == "long_line" or name == "multiple_statements":
                # 按行检测
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        # 排除注释和字符串中的误报
                        if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                            continue
                        antipatterns.append((f"第{line_num}行: {line.strip()[:50]}...", description))
                continue
            
            # 常规模式匹配
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_start = code[:match.start()].count("\n") + 1
                line = (
                    lines[line_start - 1].strip()
                    if line_start <= len(lines)
                    else ""
                )
                
                # 排除误报
                if AntiPatternDetector._should_skip_match(name, line, code, match):
                    continue
                    
                antipatterns.append((f"第{line_start}行: {line[:50]}...", description))

        return antipatterns
    
    @staticmethod
    def _detect_hardcoded_paths(lines: List[str], antipatterns: List[Tuple[str, str]]):
        """检测硬编码路径，排除误报"""
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 排除明显的非路径情况
            if (
                line.startswith('#') or  # 注释
                'http://' in line or 'https://' in line or  # URL
                'path(' in line.lower() or  # Django URL模式
                'url(' in line.lower() or   # URL模式
                '用户名' in line or '邮箱' in line or '手机号' in line or  # 中文文本
                line.startswith('"""') or line.startswith("'''") or  # 文档字符串
                'docs.djangoproject.com' in line or  # Django文档
                'www.' in line or '.com' in line or '.org' in line  # 网址
            ):
                continue
            
            # 检测真正的硬编码路径
            path_patterns = [
                r"(['\"])/(?:usr|home|var|etc|opt|tmp|root)/[\w/.-]+['\"]?",  # Unix路径
                r"(['\"])[A-Za-z]:\\[\\\w\s.-]+['\"]?",  # Windows路径
                r"\\\\[\w.-]+\\[\w\\.-]+",  # UNC路径
            ]
            
            for pattern in path_patterns:
                if re.search(pattern, line):
                    antipatterns.append((f"第{line_num}行: {line[:50]}...", "硬编码文件路径"))
                    break
    
    @staticmethod
    def _should_skip_match(pattern_name: str, line: str, code: str, match) -> bool:
        """判断是否应该跳过该匹配"""
        if pattern_name == "print_debugging":
            # 排除非调试用途的print
            if (
                'def ' in line and 'print(' in line or  # 函数定义中的print
                '"""' in line or "'''" in line or  # 文档字符串中的print
                line.strip().startswith('#') or  # 注释中的print
                'format' in line or 'f"' in line  # 格式化字符串
            ):
                return True
                
        return False


class NamingConventionChecker:
    """Python命名规范检查器"""

    SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
    PASCAL_CASE_PATTERN = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
    CAMEL_CASE_PATTERN = re.compile(r"^[a-z][a-zA-Z0-9]*$")

    @staticmethod
    def check_class_name(name: str) -> bool:
        """检查类名是否符合PEP8（PascalCase）"""
        return bool(NamingConventionChecker.PASCAL_CASE_PATTERN.match(name))

    @staticmethod
    def check_function_name(name: str) -> bool:
        """检查函数名是否符合PEP8（snake_case）

        包括以下合法命名规则：
        - 普通函数/方法：使用snake_case (如 my_function)
        - 私有函数/方法：单下划线前缀加snake_case (如 _private_method)
        - 魔术方法：双下划线前缀和后缀 (如 __init__)
        - 私有特殊方法：双下划线前缀但非魔术方法 (如 __private_method)
        """
        # 魔术方法（前后双下划线）是有效的命名
        if name.startswith("__") and name.endswith("__"):
            return True

        # 私有方法（以单下划线开头）是有效的命名
        if name.startswith("_") and not name.startswith("__"):
            # 检查下划线后面的部分是否符合snake_case
            return len(name) > 1 and (
                NamingConventionChecker.SNAKE_CASE_PATTERN.match(name[1:])
                or name[1:].islower()  # 单个字母也可以
            )

        # 类内私有方法（以双下划线开头但不以双下划线结尾）是有效的命名
        if name.startswith("__") and not name.endswith("__"):
            # 检查双下划线后面的部分是否符合snake_case
            return len(name) > 2 and (
                NamingConventionChecker.SNAKE_CASE_PATTERN.match(name[2:])
                or name[2:].islower()  # 单个字母也可以
            )

        # 常规函数名检查
        return bool(NamingConventionChecker.SNAKE_CASE_PATTERN.match(name))

    @staticmethod
    def check_variable_name(name: str) -> bool:
        """检查变量名是否符合PEP8（snake_case）

        包括以下合法命名规则：
        - 普通变量：使用snake_case (如 my_var)
        - 私有变量：单下划线前缀加snake_case (如 _private_var)
        - 类内私有变量：双下划线前缀 (如 __private_var)
        - 常量：全大写加下划线 (如 MAX_VALUE)
        - 魔术变量：双下划线前缀和后缀 (如 __all__)
        """
        # 常量使用全大写
        if name.isupper():
            return True

        # 魔术变量（前后双下划线）是有效的命名
        if name.startswith("__") and name.endswith("__"):
            return True

        # 私有变量（以单下划线开头）是有效的命名
        if name.startswith("_") and not name.startswith("__"):
            # 空名称后缀不是有效名称
            if len(name) <= 1:
                return False

            # 检查下划线后面的部分是否符合snake_case
            remainder = name[1:]
            return (
                NamingConventionChecker.SNAKE_CASE_PATTERN.match(remainder)
                or remainder.islower()  # 单个字母也可以
            )

        # 类内私有变量（以双下划线开头）是有效的命名
        if name.startswith("__"):
            # 空名称后缀不是有效名称
            if len(name) <= 2:
                return False

            # 检查双下划线后面的部分是否符合snake_case
            remainder = name[2:]
            return (
                NamingConventionChecker.SNAKE_CASE_PATTERN.match(remainder)
                or remainder.islower()  # 单个字母也可以
            )

        # 常规变量名检查
        return bool(NamingConventionChecker.SNAKE_CASE_PATTERN.match(name))

    @staticmethod
    def check_constant_name(name: str) -> bool:
        """检查常量名是否符合PEP8（大写加下划线）"""
        return name.isupper()

    @staticmethod
    def get_name_case_type(name: str) -> str:
        """识别名称的命名风格"""
        if name.startswith("_") and not name.startswith("__"):
            # 私有成员（单下划线前缀）
            remainder = name[1:]
            if (
                NamingConventionChecker.SNAKE_CASE_PATTERN.match(remainder)
                or remainder.islower()
            ):
                return "私有成员(单下划线前缀)"
            else:
                return f"私有成员(命名不规范)"

        if name.startswith("__") and name.endswith("__"):
            # 魔术方法/变量
            return "魔术方法/变量"

        if name.startswith("__"):
            # 类内私有成员（双下划线前缀）
            remainder = name[2:]
            if (
                NamingConventionChecker.SNAKE_CASE_PATTERN.match(remainder)
                or remainder.islower()
            ):
                return "类内私有成员(双下划线前缀)"
            else:
                return f"类内私有成员(命名不规范)"

        if name.isupper():
            # 常量
            return "常量(全大写)"

        if NamingConventionChecker.SNAKE_CASE_PATTERN.match(name):
            return "snake_case"
        elif NamingConventionChecker.PASCAL_CASE_PATTERN.match(name):
            return "PascalCase"
        elif NamingConventionChecker.CAMEL_CASE_PATTERN.match(name):
            return "camelCase"
        else:
            return "其他"


class QualityScoreCalculator:
    """计算代码质量评分"""

    # 权重配置
    WEIGHTS = {
        "complexity": 0.2,  # 复杂度
        "docstrings": 0.15,  # 文档字符串
        "comments": 0.1,  # 注释
        "antipatterns": 0.25,  # 反模式
        "naming": 0.15,  # 命名规范
        "type_hints": 0.15,  # 类型提示
    }

    @staticmethod
    def calculate_file_score(stats: FileStats) -> float:
        """为文件计算质量评分（0-100）"""
        # 初始化各项分数
        scores = {}

        # 计算复杂度得分 (低复杂度 = 高分)
        if stats.function_count + stats.method_count > 0:
            complexity_per_function = stats.cyclomatic_complexity / (
                stats.function_count + stats.method_count
            )
            if complexity_per_function <= 5:
                scores["complexity"] = 100
            elif complexity_per_function <= 10:
                scores["complexity"] = 80
            elif complexity_per_function <= 15:
                scores["complexity"] = 60
            elif complexity_per_function <= 20:
                scores["complexity"] = 40
            else:
                scores["complexity"] = 20
        else:
            scores["complexity"] = 100  # 没有函数的文件默认为100

        # 文档字符串得分
        if stats.class_count + stats.function_count + stats.method_count > 0:
            if stats.quality.has_docstring:
                docstring_ratio = stats.docstring_lines / (
                    stats.class_count + stats.function_count + stats.method_count
                )
                if docstring_ratio >= 3:
                    scores["docstrings"] = 100
                elif docstring_ratio >= 2:
                    scores["docstrings"] = 80
                elif docstring_ratio >= 1:
                    scores["docstrings"] = 60
                else:
                    scores["docstrings"] = 40
            else:
                scores["docstrings"] = 0
        else:
            scores["docstrings"] = 100  # 没有类和函数的文件默认为100

        # 注释得分
        if stats.code_lines > 0:
            comment_ratio = stats.comment_lines / stats.code_lines
            if comment_ratio >= 0.2:
                scores["comments"] = 100
            elif comment_ratio >= 0.15:
                scores["comments"] = 80
            elif comment_ratio >= 0.1:
                scores["comments"] = 60
            elif comment_ratio >= 0.05:
                scores["comments"] = 40
            else:
                scores["comments"] = 20
        else:
            scores["comments"] = 100  # 没有代码的文件默认为100

        # 反模式得分 (少反模式 = 高分)
        if stats.quality.antipatterns_count == 0:
            scores["antipatterns"] = 100
        elif stats.quality.antipatterns_count <= 2:
            scores["antipatterns"] = 80
        elif stats.quality.antipatterns_count <= 5:
            scores["antipatterns"] = 60
        elif stats.quality.antipatterns_count <= 10:
            scores["antipatterns"] = 40
        else:
            scores["antipatterns"] = 20

        # 命名规范得分
        if (
            stats.quality.naming_stats.good_names + stats.quality.naming_stats.bad_names
            > 0
        ):
            naming_ratio = stats.quality.naming_stats.good_names / (
                stats.quality.naming_stats.good_names
                + stats.quality.naming_stats.bad_names
            )
            scores["naming"] = min(100, naming_ratio * 100)
        else:
            scores["naming"] = 100  # 没有需要检查命名的元素

        # 类型提示得分
        scores["type_hints"] = 100 if stats.quality.has_type_hints else 0

        # 计算加权总分
        weighted_score = sum(
            scores[key] * QualityScoreCalculator.WEIGHTS[key] for key in scores
        )
        return round(weighted_score, 1)

    @staticmethod
    def get_score_category(score: float) -> Tuple[str, str]:
        """获取分数等级和颜色"""
        if score >= 90:
            return "优秀", Colors.GREEN
        elif score >= 80:
            return "良好", Colors.CYAN
        elif score >= 70:
            return "一般", Colors.BLUE
        elif score >= 60:
            return "待改进", Colors.YELLOW
        else:
            return "需重构", Colors.RED


class PythonModuleAnalyzer:
    """分析Python模块（使用AST）"""

    def __init__(self, file_path: Path, logger: AnalysisLogger):
        self.file_path = file_path
        self.stats = FileStats()
        self.module_ast = None
        self.code_content = ""
        self.logger = logger

    @staticmethod
    def _is_string_constant(node) -> bool:
        """
        检查节点是否为字符串常量
        
        根据PEP8规范，函数名使用snake_case命名
        按照模块化设计原则，将AST检查逻辑独立成函数
        """
        # Python 3.8+直接使用ast.Constant
        return isinstance(node, ast.Constant) and isinstance(node.value, str)
    
    @staticmethod
    def _has_docstring(node_body) -> bool:
        """
        检查节点体是否包含文档字符串
        
        按照模块化设计原则，将文档字符串检查逻辑独立成函数
        """
        if (len(node_body) > 0 and 
            isinstance(node_body[0], ast.Expr)):
            return PythonModuleAnalyzer._is_string_constant(node_body[0].value)
        return False

    def analyze(self) -> FileStats:
        """分析Python文件并返回统计信息"""
        self.logger.debug(f"开始分析文件: {self.file_path}")
        
        if not self._read_file():
            self.logger.warning(f"无法读取文件: {self.file_path}")
            return self.stats

        # 统计基本指标
        self._count_lines()

        # 解析AST
        if not self._parse_ast():
            self.logger.warning(f"无法解析AST: {self.file_path}")
            return self.stats

        # 执行AST分析
        if self.module_ast:
            self.logger.debug(f"正在分析AST: {self.file_path}")
            self._analyze_ast()

        # 检测反模式（只在启用时计算）
        if CONFIG["detect_antipatterns"]:
            self.logger.debug(f"正在检测反模式: {self.file_path}")
            self._detect_antipatterns()

        # 计算质量评分（只在启用时计算）
        if CONFIG["calculate_quality_score"]:
            self.stats.quality_score = QualityScoreCalculator.calculate_file_score(
                self.stats
            )
            self.logger.debug(f"完成分析文件: {self.file_path}, 质量评分: {self.stats.quality_score}")
        else:
            self.logger.debug(f"完成分析文件: {self.file_path}")
            
        return self.stats

    def _build_directory_structure(self, dir_path: Path, result: Dict) -> Dict:
        """递归构建正确的目录结构"""
        rel_path = dir_path.relative_to(self.root_path)
        
        # 创建目录条目
        dir_entry = {
            "path": str(rel_path),
            "name": rel_path.name if rel_path != Path(".") else self.root_path.name,
            "type": "directory",
            "children": [],
        }

        try:
            # 获取目录内容
            items = list(dir_path.iterdir())
            
            # 分别处理子目录和文件
            for item in items:
                # 跳过排除的项目
                if item.is_dir():
                    if self._should_exclude_dir(item.name):
                        continue
                    # 递归处理子目录
                    child_dir = self._build_directory_structure(item, result)
                    if child_dir:
                        dir_entry["children"].append(child_dir)
                else:
                    if not self._should_process_file(item.name):
                        continue
                    
                    # 处理文件
                    file_entry, stats_dict = self._analyze_file(item)
                    if file_entry:
                        dir_entry["children"].append(file_entry)
                        result["stats"][file_entry["path"]] = stats_dict
                        self._update_total_stats(result["total"], stats_dict)

        except (PermissionError, OSError):
            pass

        return dir_entry if dir_entry["children"] or rel_path == Path(".") else None

    def _read_file(self) -> bool:
        """读取文件内容，成功返回True"""
        try:
            file_size = self.file_path.stat().st_size
            if file_size > CONFIG["max_file_size_mb"] * 1024 * 1024:
                self.logger.warning(f"文件过大，跳过: {self.file_path} ({file_size / (1024*1024):.1f}MB)")
                return False

            with open(self.file_path, "r", encoding="utf-8") as f:
                self.code_content = f.read()
            self.logger.debug(f"成功读取文件: {self.file_path}")
            return True
        except Exception as e:
            self.logger.error(f"读取文件失败: {self.file_path} - {str(e)}")
            return False

    def _count_lines(self):
        """
        统计文件中不同类型的行
        
        根据CONFIG中的功能开关决定是否计算详细行数
        """
        if not self.code_content:
            return
            
        # 基本行数统计（总是需要的）
        lines = self.code_content.split('\n')
        self.stats.total_lines = len(lines)
        self.stats.physical_lines = len(lines)
        
        # 简单统计
        for line in lines:
            stripped = line.strip()
            if not stripped:
                self.stats.blank_lines += 1
            elif stripped.startswith('#'):
                self.stats.comment_lines += 1
            else:
                self.stats.code_lines += 1
        
        # 详细行数分析（只在启用时计算）
        if CONFIG["calculate_detailed_lines"]:
            line_stats = LineCountAnalyzer.analyze_lines(self.code_content)
            self.stats.total_lines = line_stats['total_lines']
            self.stats.code_lines = line_stats['code_lines']
            self.stats.comment_lines = line_stats['comment_lines']
            self.stats.blank_lines = line_stats['blank_lines']
            self.stats.docstring_lines = line_stats['docstring_lines']
            self.stats.executable_lines = line_stats['executable_lines']
            self.stats.logical_lines = line_stats['logical_lines']
            self.stats.physical_lines = line_stats['physical_lines']
            
            # 日志输出详细信息
            self.logger.debug(f"[{self.file_path.name}] "
                            f"物理行: {line_stats['physical_lines']}, "
                            f"逻辑行: {line_stats['logical_lines']}, "
                            f"可执行行: {line_stats['executable_lines']}")

    def _parse_ast(self) -> bool:
        """解析文件为AST，成功返回True"""
        try:
            self.module_ast = ast.parse(self.code_content, filename=str(self.file_path))
            self.logger.debug(f"成功解析AST: {self.file_path}")
            return True
        except SyntaxError as e:
            self.logger.warning(f"AST解析失败: {self.file_path} - {str(e)}")
            return False

    def _analyze_ast(self):
        """分析AST获取各种指标
        
        根据CONFIG中的功能开关决定进行哪些计算
        """
        if not self.module_ast:
            return

        # 初始化计数器
        class_counter = 0
        function_counter = 0
        method_counter = 0
        complexity = 0
        has_docstring = False
        has_type_hints = False
        imports = {}

        # 命名规范统计（只在启用时计算）
        naming_stats = NamingStats() if CONFIG["check_naming_conventions"] else None

        # 检查模块文档字符串（只在启用时计算）
        if CONFIG["analyze_docstrings"]:
            has_docstring = self._has_docstring(self.module_ast.body)

        # 遍历AST
        for node in ast.walk(self.module_ast):
            # 统计类
            if isinstance(node, ast.ClassDef):
                class_counter += 1

                # 检查类命名（只在启用时计算）
                if CONFIG["check_naming_conventions"] and naming_stats:
                    if NamingConventionChecker.check_class_name(node.name):
                        naming_stats.good_names += 1
                        naming_stats.pascal_case_classes += 1
                    else:
                        naming_stats.bad_names += 1
                        naming_stats.non_pascal_classes += 1
                        naming_stats.naming_issues.append(
                            f"类名 '{node.name}' 不符合PascalCase规范"
                        )

                # 检查类文档字符串（只在启用时计算）
                if CONFIG["analyze_docstrings"] and self._has_docstring(node.body):
                    has_docstring = True

            # 统计函数和方法
            elif isinstance(node, ast.FunctionDef):
                # 检查函数命名（只在启用时计算）
                if CONFIG["check_naming_conventions"] and naming_stats:
                    if NamingConventionChecker.check_function_name(node.name):
                        naming_stats.good_names += 1
                    else:
                        # 跳过魔术方法
                        if not (
                            node.name.startswith("__") and node.name.endswith("__")
                        ):
                            naming_stats.bad_names += 1
                            naming_stats.naming_issues.append(
                                f"函数名 '{node.name}' 不符合snake_case规范"
                            )

                # 检查是否为方法（类内部的函数）
                is_method = False
                for parent in ast.walk(self.module_ast):
                    if isinstance(parent, ast.ClassDef) and node in parent.body:
                        is_method = True
                        method_counter += 1
                        break

                if not is_method:
                    function_counter += 1

                # 检查函数文档字符串（只在启用时计算）
                if CONFIG["analyze_docstrings"] and self._has_docstring(node.body):
                    has_docstring = True

                # 检查类型注解
                if node.returns or any(arg.annotation for arg in node.args.args):
                    has_type_hints = True

                # 计算复杂度（只在启用时计算）
                if CONFIG["calculate_complexity"]:
                    complexity += 1  # 基础复杂度
                    for inner_node in ast.walk(node):
                        if isinstance(
                            inner_node,
                            (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler),
                        ):
                            complexity += 1
                        elif isinstance(inner_node, ast.BoolOp) and isinstance(
                            inner_node.op, (ast.And, ast.Or)
                        ):
                            complexity += len(inner_node.values) - 1

            # 检查变量命名和类型注解（只在启用时计算）
            elif isinstance(node, ast.Assign) and CONFIG["check_naming_conventions"] and naming_stats:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if NamingConventionChecker.check_variable_name(target.id):
                            naming_stats.good_names += 1
                            naming_stats.snake_case_vars += 1
                        else:
                            # 跳过魔术变量和常量
                            if (
                                not (
                                    target.id.startswith("__")
                                    and target.id.endswith("__")
                                )
                                and not target.id.isupper()
                            ):
                                naming_stats.bad_names += 1
                                case_type = (
                                    NamingConventionChecker.get_name_case_type(
                                        target.id
                                    )
                                )
                                if case_type == "camelCase":
                                    naming_stats.camel_case_vars += 1
                                naming_stats.naming_issues.append(
                                    f"变量名 '{target.id}' 不符合snake_case规范，使用了{case_type}"
                                )

            # 检查带类型注解的变量
            elif isinstance(node, ast.AnnAssign) and node.annotation:
                has_type_hints = True

                if CONFIG["check_naming_conventions"] and naming_stats and isinstance(node.target, ast.Name):
                    if NamingConventionChecker.check_variable_name(node.target.id):
                        naming_stats.good_names += 1
                    else:
                        # 跳过魔术变量和常量
                        if (
                            not (
                                node.target.id.startswith("__")
                                and node.target.id.endswith("__")
                            )
                            and not node.target.id.isupper()
                        ):
                            naming_stats.bad_names += 1
                            naming_stats.naming_issues.append(
                                f"变量名 '{node.target.id}' 不符合snake_case规范"
                            )

            # 收集导入信息（只在启用时计算）
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and CONFIG["analyze_imports"]:
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module_name = name.name
                        alias = name.asname or module_name
                        imports.setdefault(module_name, []).append(alias)
                else:  # ImportFrom
                    if node.module:  # 跳过无模块的相对导入
                        for name in node.names:
                            if name.name == "*":
                                imports.setdefault(node.module, []).append("*")
                            else:
                                imported_name = name.name
                                alias = name.asname or imported_name
                                full_name = f"{node.module}.{imported_name}"
                                imports.setdefault(node.module, []).append(
                                    imported_name
                                )

        # 更新统计信息
        self.stats.class_count = class_counter
        self.stats.function_count = function_counter
        self.stats.method_count = method_counter
        self.stats.cyclomatic_complexity = complexity if CONFIG["calculate_complexity"] else 0
        self.stats.quality.has_docstring = has_docstring if CONFIG["analyze_docstrings"] else False
        self.stats.quality.has_type_hints = has_type_hints
        self.stats.import_count = sum(len(names) for names in imports.values()) if CONFIG["analyze_imports"] else 0
        self.stats.imports = imports if CONFIG["analyze_imports"] else {}
        self.stats.quality.naming_stats = naming_stats if CONFIG["check_naming_conventions"] else NamingStats()

    def _detect_antipatterns(self):
        """检测代码中的反模式"""
        if not self.code_content:
            return

        antipatterns = AntiPatternDetector.find_antipatterns(self.code_content)
        self.stats.quality.antipatterns_count = len(antipatterns)
        self.stats.quality.antipatterns = [
            f"{desc}: {line}" for line, desc in antipatterns
        ]


class PythonProjectAnalyzer:
    """分析整个Python项目目录"""

    def __init__(self, config: Dict, logger: AnalysisLogger):
        self.config = config
        self.root_path = Path(config["target_path"]).resolve()
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.start_time = time.time()
        self.logger = logger
        self.files_processed = 0
        self.total_files = 0

    def analyze(self) -> Dict:
        """分析整个项目

        返回包含分析结果的字典
        """
        self.logger.info(f"开始分析项目: {self.root_path}")

        result = {
            "structure": [],
            "stats": {},
            "modules": {},
            "dependencies": {},
            "total": {
                "files": 0,
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0,
                "docstring_lines": 0,
                "complexity": 0,
                "classes": 0,
                "functions": 0,
                "methods": 0,
                "imports": 0,
                "antipatterns": 0,
                "naming_issues": 0,
                "quality_score": 0.0,
            },
            "quality_summary": {},
        }

        # 第一步：发现所有Python模块
        self.logger.debug("正在查找所有Python模块...")
        self._discover_modules()

        # 第二步：构建正确的目录结构
        self.logger.debug("正在分析文件内容...")
        
        # 计算总文件数
        self.total_files = self._count_python_files()
        self.logger.debug(f"找到 {self.total_files} 个Python文件")

        # 修复：使用递归方式构建正确的目录结构
        root_structure = self._build_directory_structure(self.root_path, result)
        result["structure"] = [root_structure] if root_structure else []

        self.logger.debug("文件分析完成!")

        # 构建模块关系（只在启用时计算）
        if CONFIG["build_dependency_graph"]:
            self.logger.debug("正在构建模块依赖关系...")
            self._build_module_relationships()

        # 计算项目总体质量评分（只在启用时计算）
        if CONFIG["calculate_quality_score"] and result["total"]["files"] > 0:
            self.logger.debug("正在计算质量评分...")
            result["total"]["quality_score"] = self._calculate_project_score(result)

        # 生成质量摘要
        result["quality_summary"] = self._generate_quality_summary(result)

        # 添加模块和依赖关系到结果
        result["modules"] = {
            name: asdict(module) for name, module in self.modules.items()
        }
        result["dependencies"] = {
            source: list(targets) for source, targets in self.dependencies.items()
        }

        elapsed = time.time() - self.start_time
        self.logger.info(f"分析完成! 发现 {result['total']['files']} 个文件，耗时: {round(elapsed, 2)}秒")

        return result

    def _count_python_files(self) -> int:
        """计算Python文件总数"""
        count = 0
        try:
            for item in self.root_path.rglob("*.py"):
                if item.is_file() and self._should_process_file(item.name):
                    parent_dir = item.parent
                    if not any(self._should_exclude_dir(part) for part in parent_dir.parts):
                        count += 1
        except (PermissionError, OSError):
            pass
        return count

    def _build_directory_structure(self, dir_path: Path, result: Dict) -> Dict:
        """递归构建正确的目录结构"""
        rel_path = dir_path.relative_to(self.root_path)
        
        # 创建目录条目
        dir_entry = {
            "path": str(rel_path),
            "name": rel_path.name if rel_path != Path(".") else self.root_path.name,
            "type": "directory",
            "children": [],
        }

        try:
            # 获取目录内容
            items = list(dir_path.iterdir())
            
            # 分别处理子目录和文件
            for item in items:
                # 跳过排除的项目
                if item.is_dir():
                    if self._should_exclude_dir(item.name):
                        continue
                    # 递归处理子目录
                    child_dir = self._build_directory_structure(item, result)
                    if child_dir:
                        dir_entry["children"].append(child_dir)
                else:
                    if not self._should_process_file(item.name):
                        continue
                    
                    # 处理文件
                    file_entry, stats_dict = self._analyze_file(item)
                    if file_entry:
                        dir_entry["children"].append(file_entry)
                        result["stats"][file_entry["path"]] = stats_dict
                        self._update_total_stats(result["total"], stats_dict)

        except (PermissionError, OSError):
            pass

        return dir_entry if dir_entry["children"] or rel_path == Path(".") else None

    def _calculate_project_score(self, result: Dict) -> float:
        """计算项目整体质量评分"""
        if not result["stats"]:
            return 0.0

        # 计算所有文件的加权平均分
        total_lines = 0
        weighted_sum = 0

        for path, stats in result["stats"].items():
            if not isinstance(stats, dict):
                continue

            file_lines = stats.get("total_lines", 0)
            file_score = stats.get("quality_score", 0)

            if file_lines > 0 and file_score > 0:
                total_lines += file_lines
                weighted_sum += file_lines * file_score

        if total_lines == 0:
            return 0.0

        return round(weighted_sum / total_lines, 1)

    def _generate_quality_summary(self, result: Dict) -> Dict:
        """生成项目质量摘要"""
        total = result["total"]
        total_files = total["files"]

        if total_files == 0:
            return {}

        # 计算总体质量评分
        score = result["total"]["quality_score"]
        score_category, score_color = QualityScoreCalculator.get_score_category(score)

        # 计算文档覆盖率
        files_with_docstrings = 0
        for stats in result["stats"].values():
            if isinstance(stats, dict) and stats.get("quality", {}).get(
                "has_docstring", False
            ):
                files_with_docstrings += 1

        docstring_coverage = (
            round((files_with_docstrings / total_files) * 100, 1)
            if total_files > 0
            else 0
        )

        # 计算类型提示覆盖率
        files_with_type_hints = 0
        for stats in result["stats"].values():
            if isinstance(stats, dict) and stats.get("quality", {}).get(
                "has_type_hints", False
            ):
                files_with_type_hints += 1

        type_hint_coverage = (
            round((files_with_type_hints / total_files) * 100, 1)
            if total_files > 0
            else 0
        )

        # 获取平均复杂度
        avg_complexity = (
            round(total["complexity"] / (total["functions"] + total["methods"]), 1)
            if (total["functions"] + total["methods"]) > 0
            else 0
        )

        # 计算代码注释比例
        comment_ratio = (
            round(total["comment_lines"] / total["code_lines"] * 100, 1)
            if total["code_lines"] > 0
            else 0
        )

        return {
            "quality_score": score,
            "quality_category": score_category,
            "files_with_docstrings_percent": docstring_coverage,
            "files_with_type_hints_percent": type_hint_coverage,
            "average_complexity": avg_complexity,
            "comment_to_code_ratio": comment_ratio,
            "total_antipatterns": total["antipatterns"],
            "total_naming_issues": total["naming_issues"],
        }

    def _discover_modules(self):
        """第一步：发现项目中的所有Python模块"""
        for root, dirs, files in os.walk(self.root_path, topdown=True):
            # 根据排除模式过滤目录
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]

            rel_path = Path(root).relative_to(self.root_path)
            module_path = str(rel_path).replace(os.sep, ".")

            # 检查当前目录是否为包
            has_init = "__init__.py" in files
            if has_init:
                # 这个目录是一个包
                package_name = module_path if module_path != "." else ""
                self.modules[package_name] = ModuleInfo(
                    name=package_name,
                    path=str(rel_path / "__init__.py"),
                    is_package=True,
                )

            # 处理目录中的所有Python文件
            for f in files:
                if f.endswith(".py") and not self._is_excluded(
                    f, self.config["exclude_files"]
                ):
                    if f == "__init__.py":
                        continue  # 上面已经处理过了

                    # 计算模块名
                    if module_path == ".":
                        module_name = f[:-3]  # 移除.py
                    else:
                        module_name = f"{module_path}.{f[:-3]}"

                    self.modules[module_name] = ModuleInfo(
                        name=module_name, path=str(rel_path / f), is_package=False
                    )

    def _build_module_relationships(self):
        """基于导入关系构建模块之间的依赖关系"""
        # 首先，将模块路径转换为名称以便查找
        path_to_name = {mod.path: name for name, mod in self.modules.items()}

        # 对于每个模块，检查其导入项
        for module_name, module_info in self.modules.items():
            if not module_info.stats or not module_info.stats.imports:
                continue

            # 如果需要，初始化依赖跟踪
            if module_name not in self.dependencies:
                self.dependencies[module_name] = set()

            # 处理导入项
            for imported_module, imported_items in module_info.stats.imports.items():
                # 跳过标准库和第三方模块
                if imported_module in module_name or any(
                    imported_module.startswith(f"{mod_name}.")
                    for mod_name in self.modules
                ):
                    # 找到一个项目本地导入
                    self.dependencies[module_name].add(imported_module)

                    # 更新目标模块的 "imported_by"
                    if imported_module in self.modules:
                        self.modules[imported_module].imported_by.append(module_name)

    def _should_exclude_dir(self, dir_name: str) -> bool:
        """检查一个目录是否应该被排除"""
        if not self.config["show_hidden"] and dir_name.startswith("."):
            return True
        return self._is_excluded(dir_name, self.config["exclude_dirs"])

    def _should_process_file(self, file_name: str) -> bool:
        """检查一个文件是否应该被处理"""
        if not self.config["show_hidden"] and file_name.startswith("."):
            return False
        if self._is_excluded(file_name, self.config["exclude_files"]):
            return False
        return self._should_include(file_name)

    def _is_excluded(self, name: str, patterns: List[str]) -> bool:
        """检查一个名称是否匹配任何排除模式"""
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    def _should_include(self, name: str) -> bool:
        """检查一个名称是否匹配任何包含模式"""
        if not self.config["include_patterns"]:
            return True
        return any(fnmatch.fnmatch(name, p) for p in self.config["include_patterns"])

    def _create_dir_entry(self, rel_path: Path) -> Dict:
        """为结果结构创建一个目录条目"""
        return {
            "path": str(rel_path),
            "name": rel_path.name if rel_path != Path(".") else self.root_path.name,
            "type": "directory",
            "children": [],
        }

    def _analyze_file(self, file_path: Path) -> Tuple[Dict, Dict]:
        """分析单个Python文件

        返回一个元组(file_entry, stats_dict)
        """
        try:
            # 更新进度计数器
            self.files_processed += 1
            
        # 显示文件处理进度（只在DEBUG模式下显示）
            self.logger.file_progress(
                str(file_path.relative_to(self.root_path)),
                self.files_processed,
                self.total_files
            )
            
            rel_path = str(file_path.relative_to(self.root_path))

            # 为结构创建文件条目
            file_entry = {
                "path": rel_path,
                "name": file_path.name,
                "type": "file",
                "size": file_path.stat().st_size,
            }

            # 跳过非Python文件或太大的文件
            if (
                not file_path.suffix == ".py"
                or file_path.stat().st_size
                > self.config["max_file_size_mb"] * 1024 * 1024
            ):
                return file_entry, {"total_lines": 0}

            # 分析Python文件
            analyzer = PythonModuleAnalyzer(file_path, self.logger)
            stats = analyzer.analyze()
            stats_dict = asdict(stats)

            # 使用统计信息更新模块信息
            module_rel_path = rel_path.replace(os.sep, "/")
            for name, module in self.modules.items():
                if module.path == module_rel_path:
                    module.stats = stats
                    break

            return file_entry, stats_dict
        except Exception as e:
            self.logger.error(f"分析文件 {file_path} 时出错: {e}")
            # 即使分析失败也返回条目
            return {
                "path": str(file_path.relative_to(self.root_path)),
                "name": file_path.name,
                "type": "file",
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "error": str(e),
            }, {"total_lines": 0}

    def _update_total_stats(self, total: Dict, stats: Dict):
        """
        使用文件统计信息更新总体统计信息
        
        按照模块化设计原则，专门负责统计数据聚合
        遵循PEP8规范，提供完整的行数统计
        """
        total["files"] += 1
        total["total_lines"] += stats.get("total_lines", 0)
        total["code_lines"] += stats.get("code_lines", 0)
        total["comment_lines"] += stats.get("comment_lines", 0)
        total["blank_lines"] += stats.get("blank_lines", 0)
        total["docstring_lines"] += stats.get("docstring_lines", 0)
        total["complexity"] += stats.get("cyclomatic_complexity", 0)
        total["classes"] += stats.get("class_count", 0)
        total["functions"] += stats.get("function_count", 0)
        total["methods"] += stats.get("method_count", 0)
        total["imports"] += stats.get("import_count", 0)
        
        # 新增：详细行数统计
        total["executable_lines"] = total.get("executable_lines", 0) + stats.get("executable_lines", 0)
        total["logical_lines"] = total.get("logical_lines", 0) + stats.get("logical_lines", 0)
        total["physical_lines"] = total.get("physical_lines", 0) + stats.get("physical_lines", 0)

        quality = stats.get("quality", {})
        if isinstance(quality, dict):
            if "antipatterns_count" in quality:
                total["antipatterns"] += quality["antipatterns_count"]

            naming_stats = quality.get("naming_stats", {})
            if isinstance(naming_stats, dict) and "naming_issues" in naming_stats:
                total["naming_issues"] += len(naming_stats["naming_issues"])


class DependencyVisualizer:
    """使用ASCII字符创建简单的依赖关系可视化"""

    @staticmethod
    def create_dependency_matrix(
        dependencies: Dict[str, List[str]], modules: List[str]
    ) -> List[List[str]]:
        """创建一个依赖关系矩阵

        返回一个二维数组，每个单元格包含依赖类型的符号
        """
        # 准备模块列表（按字母排序）
        sorted_modules = sorted(modules)

        # 创建空矩阵
        matrix = []
        for i in range(len(sorted_modules)):
            row = []
            for j in range(len(sorted_modules)):
                row.append(" ")
            matrix.append(row)

        # 填充矩阵
        for i, source in enumerate(sorted_modules):
            for j, target in enumerate(sorted_modules):
                if i == j:
                    matrix[i][j] = "X"  # 自引用
                elif source in dependencies and target in dependencies[source]:
                    matrix[i][j] = "●"  # 依赖

        return matrix, sorted_modules

    @staticmethod
    def generate_ascii_dependency_graph(dependencies: Dict[str, List[str]]) -> str:
        """生成ASCII依赖关系图

        返回一个字符串表示的依赖图
        """
        if not dependencies:
            return "没有发现依赖关系"

        # 获取所有模块
        all_modules = set(dependencies.keys())
        for deps in dependencies.values():
            all_modules.update(deps)

        # 过滤掉标准库和外部库
        project_modules = [mod for mod in all_modules if "." in mod or len(mod) < 15]

        # 如果模块太多，只显示顶级模块
        if len(project_modules) > 20:
            top_modules = set()
            for mod in project_modules:
                top_mod = mod.split(".")[0]
                top_modules.add(top_mod)

            # 重建依赖关系
            top_dependencies = {}
            for source, targets in dependencies.items():
                source_top = source.split(".")[0]
                if source_top not in top_dependencies:
                    top_dependencies[source_top] = set()

                for target in targets:
                    target_top = target.split(".")[0]
                    if source_top != target_top:
                        top_dependencies[source_top].add(target_top)

            # 转换为列表
            for source, targets in top_dependencies.items():
                top_dependencies[source] = list(targets)

            dependencies = top_dependencies
            project_modules = list(top_modules)

        # 针对小型项目的简单依赖图
        if len(project_modules) <= 10:
            return DependencyVisualizer._generate_simple_graph(
                dependencies, project_modules
            )

        # 针对大型项目的矩阵依赖图
        return DependencyVisualizer._generate_matrix_graph(
            dependencies, project_modules
        )

    @staticmethod
    def _generate_simple_graph(
        dependencies: Dict[str, List[str]], modules: List[str]
    ) -> str:
        """为项目生成简单的节点和边树形图"""
        lines = ["依赖关系图（每个模块依赖的其他模块）:"]

        # 按有依赖关系的模块排序
        sorted_modules = sorted(
            [m for m in modules if m in dependencies and dependencies.get(m)]
        )

        # 如果没有依赖关系，添加提示
        if not sorted_modules:
            lines.append("未发现模块间依赖关系")
            return "\n".join(lines)

        # 为每个模块生成其依赖树
        for source in sorted_modules:
            deps = dependencies.get(source, [])
            if deps:
                lines.append(f"{source} 依赖:")
                for i, target in enumerate(sorted(deps)):
                    if i == len(deps) - 1:
                        lines.append(f"  └── {target}")
                    else:
                        lines.append(f"  ├── {target}")

        return "\n".join(lines)

    @staticmethod
    def _generate_matrix_graph(
        dependencies: Dict[str, List[str]], modules: List[str]
    ) -> str:
        """为大型项目生成依赖关系矩阵"""
        matrix, sorted_modules = DependencyVisualizer.create_dependency_matrix(
            dependencies, modules
        )

        lines = ["依赖关系矩阵:"]

        # 创建列标题（使用缩写）
        max_name_length = min(10, max(len(mod) for mod in sorted_modules))
        header = "    "  # 左侧空间

        # 添加列标签
        for j, mod in enumerate(sorted_modules):
            name = mod[:max_name_length]
            header += f"{j:02d} "

        lines.append(header)
        lines.append("    " + "-" * (len(sorted_modules) * 3))

        # 添加矩阵行
        for i, mod in enumerate(sorted_modules):
            name = mod[:max_name_length]
            row = f"{i:02d}| "

            for j in range(len(sorted_modules)):
                row += f" {matrix[i][j]} "

            lines.append(row)

        # 添加图例
        lines.append("")
        lines.append("图例:")
        lines.append("X = 自引用")
        lines.append("● = 依赖关系")

        # 添加模块索引
        lines.append("")
        lines.append("模块索引:")
        for i, mod in enumerate(sorted_modules):
            lines.append(f"{i:02d} = {mod}")

        return "\n".join(lines)


class OutputFormatter:
    """
    格式化分析结果输出
    
    按照模块化设计原则，专门负责结果的格式化和输出
    遵循PEP8命名规范，提供丰富的输出选项
    """

    @staticmethod
    def format(result: Dict, config: Dict) -> str:
        """根据配置的输出格式格式化结果"""
        formatters = {
            "text": OutputFormatter._text_format,
            "markdown": OutputFormatter._markdown_format,
            "json": OutputFormatter._json_format,
        }
        return formatters[config["output_format"]](result, config)

    @staticmethod
    def _text_format(result: Dict, config: Dict) -> str:
        """格式化结果为纯文本"""
        output = [
            f"{Colors.BOLD}{Colors.CYAN}" + "=" * 50 + f"{Colors.RESET}",
            f"{Colors.BOLD}{Colors.CYAN}=         Python代码库分析报告 v{__version__}            ={Colors.RESET}",
            f"{Colors.BOLD}{Colors.CYAN}" + "=" * 50 + f"{Colors.RESET}",
            ""
        ]

        # 项目质量评分
        output.append(f"{Colors.BOLD}{Colors.BLUE}=== 项目质量评分 ==={Colors.RESET}")
        summary = result.get("quality_summary", {})
        if summary:
            score = summary.get("quality_score", 0)
            category = summary.get("quality_category", "未知")
            category_color = QualityScoreCalculator.get_score_category(score)[1]

            output.append(
                f"总体质量: {category_color}{score} - {category}{Colors.RESET}"
            )
            output.append(
                f"文档覆盖率: {summary.get('files_with_docstrings_percent', 0)}%"
            )
            output.append(
                f"类型提示覆盖率: {summary.get('files_with_type_hints_percent', 0)}%"
            )
            output.append(f"平均复杂度: {summary.get('average_complexity', 0)}")
            output.append(f"注释比例: {summary.get('comment_to_code_ratio', 0)}%")
            output.append(f"反模式问题: {summary.get('total_antipatterns', 0)}")
            output.append(f"命名规范问题: {summary.get('total_naming_issues', 0)}")
            output.append("")

        # 项目摘要
        output.append(f"{Colors.BOLD}{Colors.BLUE}=== 项目摘要 ==={Colors.RESET}")
        total = result["total"]
        output.append(f"分析的文件数: {total['files']}")
        
        # 详细行数统计
        if config.get("show_line_numbers", True):
            output.append(f"总行数: {total['total_lines']:,}")
            output.append(
                f"代码行: {total['code_lines']:,} ({OutputFormatter._percentage(total['code_lines'], total['total_lines'])}%)"
            )
            output.append(
                f"注释行: {total['comment_lines']:,} ({OutputFormatter._percentage(total['comment_lines'], total['total_lines'])}%)"
            )
            output.append(
                f"文档字符串行: {total['docstring_lines']:,} ({OutputFormatter._percentage(total['docstring_lines'], total['total_lines'])}%)"
            )
            output.append(
                f"空行: {total['blank_lines']:,} ({OutputFormatter._percentage(total['blank_lines'], total['total_lines'])}%)"
            )
            
            # 新增：详细行数分析
            if total.get('executable_lines', 0) > 0:
                output.append(f"可执行代码行: {total['executable_lines']:,}")
            if total.get('logical_lines', 0) > 0:
                output.append(f"逻辑代码行: {total['logical_lines']:,}")
            if total.get('physical_lines', 0) > 0:
                output.append(f"物理代码行: {total['physical_lines']:,}")
        else:
            output.append(f"总行数: {total['total_lines']}")
            output.append(
                f"代码行: {total['code_lines']} ({OutputFormatter._percentage(total['code_lines'], total['total_lines'])}%)"
            )
            output.append(
                f"注释行: {total['comment_lines']} ({OutputFormatter._percentage(total['comment_lines'], total['total_lines'])}%)"
            )
            output.append(
                f"文档字符串行: {total['docstring_lines']} ({OutputFormatter._percentage(total['docstring_lines'], total['total_lines'])}%)"
            )
            output.append(
                f"空行: {total['blank_lines']} ({OutputFormatter._percentage(total['blank_lines'], total['total_lines'])}%)"
            )
        output.append(f"类数量: {total['classes']}")
        output.append(f"函数数量: {total['functions']}")
        output.append(f"方法数量: {total['methods']}")
        output.append(f"总复杂度: {total['complexity']}")
        output.append(f"检测到的反模式: {total['antipatterns']}")
        output.append(f"命名规范问题: {total['naming_issues']}")
        output.append("")

        # 项目结构
        output.append(f"{Colors.BOLD}{Colors.BLUE}=== 项目结构 ==={Colors.RESET}")
        for entry in result["structure"]:
            OutputFormatter._format_entry(entry, result, output, 0)

        # 模块依赖关系可视化
        if config["analyze_imports"] and result["dependencies"]:
            output.append(
                f"\n{Colors.BOLD}{Colors.BLUE}=== 模块依赖关系 ==={Colors.RESET}"
            )

            # 生成ASCII依赖图
            dependency_graph = DependencyVisualizer.generate_ascii_dependency_graph(
                result["dependencies"]
            )
            output.append(dependency_graph)

        # 最复杂的文件
        if result["stats"]:
            output.append(
                f"\n{Colors.BOLD}{Colors.BLUE}=== 前10个最复杂的文件 ==={Colors.RESET}"
            )
            complex_files = sorted(
                [
                    (
                        path,
                        stats.get("cyclomatic_complexity", 0),
                        stats.get("quality_score", 0),
                    )
                    for path, stats in result["stats"].items()
                    if isinstance(stats, dict)
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            for path, complexity, score in complex_files:
                score_color = QualityScoreCalculator.get_score_category(score)[1]
                output.append(
                    f"{path}: 复杂度 {complexity}, 质量评分: {score_color}{score}{Colors.RESET}"
                )

        # 反模式摘要
        if config["detect_antipatterns"]:
            antipattern_files = []
            for path, stats in result["stats"].items():
                if not isinstance(stats, dict):
                    continue
                quality = stats.get("quality", {})
                if (
                    isinstance(quality, dict)
                    and quality.get("antipatterns_count", 0) > 0
                ):
                    antipattern_files.append(
                        (
                            path,
                            quality["antipatterns_count"],
                            quality.get("antipatterns", []),
                        )
                    )

            if antipattern_files:
                output.append(
                    f"\n{Colors.BOLD}{Colors.YELLOW}=== 含有反模式的文件 ==={Colors.RESET}"
                )
                for path, count, patterns in sorted(
                    antipattern_files, key=lambda x: x[1], reverse=True
                )[:10]:
                    output.append(f"{path}: {count} 个反模式")
                    for pattern in patterns[:3]:  # 只显示前3个
                        output.append(f"  - {pattern}")
                    if len(patterns) > 3:
                        output.append(f"  - ... 还有 {len(patterns) - 3} 个")

        # 命名规范问题
        if config["check_naming_conventions"]:
            naming_issue_files = []
            for path, stats in result["stats"].items():
                if not isinstance(stats, dict):
                    continue
                quality = stats.get("quality", {})
                if not isinstance(quality, dict):
                    continue
                naming_stats = quality.get("naming_stats", {})
                if (
                    isinstance(naming_stats, dict)
                    and naming_stats.get("bad_names", 0) > 0
                ):
                    naming_issue_files.append(
                        (
                            path,
                            naming_stats["bad_names"],
                            naming_stats.get("naming_issues", []),
                        )
                    )

            if naming_issue_files:
                output.append(
                    f"\n{Colors.BOLD}{Colors.YELLOW}=== 命名规范问题 ==={Colors.RESET}"
                )
                for path, count, issues in sorted(
                    naming_issue_files, key=lambda x: x[1], reverse=True
                )[:10]:
                    output.append(f"{path}: {count} 个命名规范问题")
                    for issue in issues[:3]:  # 只显示前3个
                        output.append(f"  - {issue}")
                    if len(issues) > 3:
                        output.append(f"  - ... 还有 {len(issues) - 3} 个")

        return "\n".join(output)

    @staticmethod
    def _percentage(part, total):
        """计算百分比，保留1位小数"""
        return round((part / total) * 100, 1) if total else 0

    @staticmethod
    def _format_entry(entry: Dict, result: Dict, output: List, level: int):
        """格式化目录或文件条目为文本输出"""
        indent = "    " * level
        prefix = f"{indent}|-- "

        if entry["type"] == "directory":
            output.append(f"{prefix}{Colors.BLUE}{entry['name']}/{Colors.RESET}")
            # 修复：正确处理子目录的递归显示
            for child in sorted(
                entry.get("children", []),
                key=lambda x: (x["type"] != "directory", x["name"]),
            ):
                OutputFormatter._format_entry(child, result, output, level + 1)
        else:
            file_info = OutputFormatter._file_info(entry, result)
            if entry["name"].endswith(".py"):
                output.append(
                    f"{prefix}{Colors.CYAN}{entry['name']}{Colors.RESET} {file_info}"
                )
            else:
                output.append(f"{prefix}{entry['name']} {file_info}")

    @staticmethod
    def _file_info(entry: Dict, result: Dict) -> str:
        """
        格式化文件信息用于显示
        
        按照模块化设计原则，提供详细的文件统计信息
        遵循PEP8规范，返回格式化的文件信息字符串
        """
        stats = result["stats"].get(entry["path"])
        if not isinstance(stats, dict):
            return ""

        info = []
        score = stats.get("quality_score", 0)

        if score > 0:
            score_category, score_color = QualityScoreCalculator.get_score_category(
                score
            )
            if CONFIG.get("color_output", True):
                info.append(f"质量: {score_color}{score}{Colors.RESET}")
            else:
                info.append(f"质量: {score}")

        # 详细行数统计
        if CONFIG.get("show_line_numbers", True) and stats.get("total_lines", 0) > 0:
            total_lines = stats["total_lines"]
            
            # 基本行数信息
            line_info = [f"{total_lines}行"]
            
            # 添加详细行数分解
            if stats.get("code_lines", 0) > 0:
                code_lines = stats["code_lines"]
                percentage = round((code_lines / total_lines) * 100, 1) if total_lines > 0 else 0
                line_info.append(f"{code_lines}代码({percentage}%)")
            
            if stats.get("comment_lines", 0) > 0:
                line_info.append(f"{stats['comment_lines']}注释")
                
            if stats.get("docstring_lines", 0) > 0:
                line_info.append(f"{stats['docstring_lines']}文档")
            
            # 将行数信息添加到info中
            info.append(", ".join(line_info))
        elif stats.get("total_lines", 0) > 0:
            # 简化模式，只显示总行数
            info.append(f"{stats['total_lines']}行")

        # 代码结构信息
        if "class_count" in stats and stats["class_count"] > 0:
            info.append(f"{stats['class_count']}类")

        if "function_count" in stats and stats["function_count"] > 0:
            info.append(f"{stats['function_count']}函数")

        if "method_count" in stats and stats["method_count"] > 0:
            info.append(f"{stats['method_count']}方法")

        # 复杂度信息（使用颜色编码）
        if "cyclomatic_complexity" in stats and stats["cyclomatic_complexity"] > 0:
            complexity = stats["cyclomatic_complexity"]
            if CONFIG.get("color_output", True):
                if complexity > 30:
                    info.append(f"复杂度: {Colors.RED}{complexity}{Colors.RESET}")
                elif complexity > 15:
                    info.append(f"复杂度: {Colors.YELLOW}{complexity}{Colors.RESET}")
                else:
                    info.append(f"复杂度: {complexity}")
            else:
                info.append(f"复杂度: {complexity}")

        # 质量问题信息
        quality = stats.get("quality", {})
        if isinstance(quality, dict):
            if quality.get("antipatterns_count", 0) > 0:
                count = quality["antipatterns_count"]
                if CONFIG.get("color_output", True):
                    info.append(f"{Colors.YELLOW}{count}个反模式{Colors.RESET}")
                else:
                    info.append(f"{count}个反模式")

            naming_stats = quality.get("naming_stats", {})
            if (
                isinstance(naming_stats, dict)
                and naming_stats.get("bad_names", 0) > 0
            ):
                count = naming_stats["bad_names"]
                if CONFIG.get("color_output", True):
                    info.append(f"{Colors.YELLOW}{count}个命名问题{Colors.RESET}")
                else:
                    info.append(f"{count}个命名问题")

        # 文件大小信息
        if entry.get("size"):
            info.append(f"{OutputFormatter._format_size(entry['size'])}")

        return f"({', '.join(info)})" if info else ""

    @staticmethod
    def _format_size(size: int) -> str:
        """格式化文件大小使其易于阅读"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

    @staticmethod
    def _markdown_format(result: Dict, config: Dict) -> str:
        """格式化结果为Markdown"""
        config_no_color = config.copy()
        config_no_color["color_output"] = False

        # 生成没有颜色代码的纯文本
        Colors.disable()
        text = OutputFormatter._text_format(result, config_no_color)

        # 恢复颜色支持
        if config["color_output"] and Colors.supports_color():
            Colors.RESET = "\033[0m"
            Colors.BOLD = "\033[1m"
            Colors.RED = "\033[31m"
            Colors.GREEN = "\033[32m"
            Colors.YELLOW = "\033[33m"
            Colors.BLUE = "\033[34m"
            Colors.MAGENTA = "\033[35m"
            Colors.CYAN = "\033[36m"
            Colors.GRAY = "\033[90m"

        return f"# Python项目分析报告\n\n```\n{text}\n```"

    @staticmethod
    def _json_format(result: Dict, config: Dict) -> str:
        """格式化结果为JSON"""
        return json.dumps(result, indent=2, ensure_ascii=False)


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Python代码库分析工具 v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  {os.path.basename(sys.argv[0])} .                    # 分析当前目录
  {os.path.basename(sys.argv[0])} /path/to/project     # 分析指定项目
  {os.path.basename(sys.argv[0])} . --debug           # 启用调试模式
  {os.path.basename(sys.argv[0])} . --format json     # 输出JSON格式
        """
    )
    parser.add_argument(
        "path", nargs="?", default=".", help="Python项目路径（默认：当前目录）"
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="输出格式",
    )
    
    # 日志级别控制（互斥组）
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--debug", "-d", action="store_true", 
        help="启用调试模式（显示详细日志）"
    )
    log_group.add_argument(
        "--quiet", "-q", action="store_true", 
        help="安静模式（只显示警告和错误）"
    )
    
    # 功能开关（控制功能的计算和输出）
    parser.add_argument(
        "--no-detailed-lines", action="store_true", 
        help="不计算详细行数分解"
    )
    parser.add_argument(
        "--no-antipatterns", action="store_true", 
        help="不检测反模式"
    )
    parser.add_argument(
        "--no-naming", action="store_true", 
        help="不检查命名规范"
    )
    parser.add_argument(
        "--no-imports", action="store_true", 
        help="不分析导入依赖"
    )
    parser.add_argument(
        "--no-complexity", action="store_true", 
        help="不计算循环复杂度"
    )
    parser.add_argument(
        "--no-docstrings", action="store_true", 
        help="不分析文档字符串"
    )
    parser.add_argument(
        "--no-quality-score", action="store_true", 
        help="不计算质量评分"
    )
    parser.add_argument(
        "--no-dependency-graph", action="store_true", 
        help="不构建依赖关系图"
    )
    parser.add_argument(
        "--no-project-structure", action="store_true", 
        help="不显示项目结构"
    )
    
    parser.add_argument(
        "--version", "-v", action="version", 
        version=f"Python代码库分析工具 v{__version__}"
    )
    
    # 其他参数
    parser.add_argument("--max-workers", type=int, default=4, help="最大并行工作线程数")
    parser.add_argument(
        "--max-file-size", type=int, default=10, help="要分析的最大文件大小（MB）"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="禁用彩色输出"
    )
    parser.add_argument("--show-hidden", action="store_true", help="包含隐藏文件和目录")

    args = parser.parse_args()

    # 直接更新CONFIG
    CONFIG["target_path"] = args.path
    CONFIG["output_format"] = args.format
    CONFIG["max_workers"] = args.max_workers
    CONFIG["max_file_size_mb"] = args.max_file_size
    CONFIG["show_hidden"] = args.show_hidden
    CONFIG["color_output"] = not args.no_color
    
    # 设置日志级别
    if args.debug:
        CONFIG["log_level"] = "DEBUG"
    elif args.quiet:
        CONFIG["log_level"] = "WARNING"
    else:
        CONFIG["log_level"] = "INFO"
    
    # 设置功能开关
    if args.no_detailed_lines:
        CONFIG["calculate_detailed_lines"] = False
    if args.no_antipatterns:
        CONFIG["detect_antipatterns"] = False
    if args.no_naming:
        CONFIG["check_naming_conventions"] = False
    if args.no_imports:
        CONFIG["analyze_imports"] = False
    if args.no_complexity:
        CONFIG["calculate_complexity"] = False
    if args.no_docstrings:
        CONFIG["analyze_docstrings"] = False
    if args.no_quality_score:
        CONFIG["calculate_quality_score"] = False
    if args.no_dependency_graph:
        CONFIG["build_dependency_graph"] = False
    if args.no_project_structure:
        CONFIG["show_project_structure"] = False

    return args


# 在 if __name__ == "__main__": 部分进行如下修改
if __name__ == "__main__":
    try:
        # 保存用户自定义的目标路径（如果已设置）
        user_target_path = CONFIG.get("target_path", ".")

        # 解析命令行参数（直接更新CONFIG）
        args = parse_args()

        # 如果命令行未指定路径但用户在代码中已设置了路径，则恢复用户设置
        if args.path == "." and user_target_path != ".":
            CONFIG["target_path"] = user_target_path
        
        # 创建日志管理器（使用CONFIG中的日志级别）
        log_level = LogLevel(CONFIG["log_level"])
        logger = AnalysisLogger(log_level)

        # 禁用彩色输出（如果需要）
        if not CONFIG["color_output"]:
            Colors.disable()

        logger.debug(f"日志级别: {CONFIG['log_level']}")
        logger.debug(f"功能开关: 详细行数={CONFIG['calculate_detailed_lines']}, "
                    f"反模式={CONFIG['detect_antipatterns']}, "
                    f"命名检查={CONFIG['check_naming_conventions']}, "
                    f"质量评分={CONFIG['calculate_quality_score']}")

        # 运行分析
        analyzer = PythonProjectAnalyzer(CONFIG, logger)
        result = analyzer.analyze()

        # 输出结果
        output = OutputFormatter.format(result, CONFIG)
        print(output)
        
    except KeyboardInterrupt:
        logger.warning("\n操作被用户取消")
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        if CONFIG["log_level"] == "DEBUG":
            import traceback
            traceback.print_exc()
