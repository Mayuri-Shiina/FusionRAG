import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# -------------------------- 第一步：加载.env配置文件 --------------------------
load_dotenv()

# -------------------------- 第二步：读取.env配置（带默认值，防止配置缺失） --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# -------------------------- 第三步：定义日志路径（自动推导项目根） --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# -------------------------- 第四步：定义日志格式（彩色、结构化、易读） --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# -------------------------- 第五步：初始化日志配置（核心方法） --------------------------
def init_logger():
    """
    初始化全局日志配置
    1. 移除loguru默认控制台输出（避免重复打印）
    2. 根据.env配置开启/关闭控制台输出
    3. 根据.env配置开启/关闭文件输出（自动创建logs文件夹）
    4. 配置日志格式、级别、分割、保留策略
    :return: 配置完成的loguru logger实例
    """
    # 1. 移除loguru默认的控制台输出
    logger.remove()

    # 2. 配置控制台输出（若.env开启）
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )

    # 3. 配置文件输出（若.env开启）
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    return logger

# -------------------------- 第六步：初始化并终极修正全局logger --------------------------
base_logger = init_logger()

def fix_log_position(record):
    """遍历调用栈，跳过loguru内部帧+工具类自身帧，提取业务代码实际调用位置"""
    for frame in inspect.stack():
        # 终极过滤：排除loguru内部 + 排除工具类logger.py自身，直接定位业务模块
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # 更新日志字段为业务代码实际位置
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# 应用终极修复，导出全局可用的logger
logger = base_logger.patch(fix_log_position)
