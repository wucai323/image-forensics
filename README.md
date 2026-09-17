# 图像取证助手（Image Forensics）

基于启发式规则的桌面小工具，用于辅助判断一张图片是否可能经过数字加工（重保存、局部编辑、元数据篡改等）。

> **重要**：本工具**不能**给出 100% 确定结论，**不能**替代司法鉴定或专业取证报告。结果仅供参考。

## 功能

1. **打开图像**（文件选择对话框）→ **开始分析**
2. 输出：
   - 结论标签：`可能原图` / `可能被加工` / `不确定`
   - 置信度与综合风险分
   - 分项详情与中文说明
   - 原图与 **ELA（误差级分析）** 并排预览
3. 检测项（均为启发式评分）：
   - **EXIF / 元数据**：编辑软件痕迹、缺少相机字段、时间戳异常（Pillow）
   - **ELA**：JPEG 重压缩差异热力图与异常分
   - **噪声 / 残差**：分块高频残差一致性

## 环境要求

- Windows 10/11（开发与打包目标平台）
- Python 3.10+（推荐 3.11 / 3.12；亦支持 3.13）
- 依赖见 `requirements.txt`（已固定版本：Pillow / numpy / customtkinter）
- Windows 需使用带 **tcl/tk** 的官方 Python 安装包（GUI 依赖 tkinter）

在 Linux 上可无界面运行分析引擎与 `scripts/self_check.py`；GUI 需要可用的显示环境。

## 安装（Windows）

```bat
cd image-forensics
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

```bat
python main.py
```

或：

```bat
python -m app
```

## 自检（无界面）

生成「干净 JPEG」与「轻度编辑/重保存 JPEG」，比较综合风险分：

```bat
python scripts/self_check.py
```

成功时退出码为 `0`，并打印两张样张的分项得分。

也可：

```bat
python -m unittest tests.test_analysis_smoke -v
```

## 使用 PyInstaller 打包（Windows）

在已激活的虚拟环境中：

```bat
pip install pyinstaller==6.10.0
pyinstaller --noconfirm --windowed --name ImageForensics ^
  --collect-all customtkinter ^
  --hidden-import PIL._tkinter_finder ^
  main.py
```

产物一般在 `dist\ImageForensics\ImageForensics.exe`。

若 `customtkinter` 主题资源缺失，可补充：

```bat
pyinstaller --noconfirm --windowed --name ImageForensics ^
  --collect-all customtkinter --collect-all tkinterdnd2 ^
  main.py
```

（本项目未强制依赖 `tkinterdnd2`；拖放为可选项。）

## 项目结构

```
image-forensics/
  README.md
  requirements.txt
  main.py                 # 入口
  app/
    __main__.py           # python -m app
    analysis/             # 与 UI 分离的检测引擎
      engine.py
      exif_check.py
      ela_check.py
      noise_check.py
    ui/
      main_window.py      # customtkinter / tkinter GUI
  scripts/
    self_check.py
  tests/
    test_analysis_smoke.py
```

## 方法局限（请务必阅读）

| 局限 | 说明 |
|------|------|
| 非司法级 | 无机器学习模型训练语料背书，亦无相机指纹（PRNU）等专业手段 |
| 社交平台转码 | 微信/微博等会剥离 EXIF 并重压缩，易被判为「不确定」或偏加工 |
| PNG / 截图 | 本身常无相机 EXIF；ELA 对非 JPEG 参考价值有限 |
| 生成式 AI | 未专门检测扩散模型伪影；噪声项仅作弱信号 |
| 误报 / 漏报 | 重度锐化、美颜、滤镜可能抬高风险；精心处理的伪造也可能偏低 |
| 置信度含义 | 表示「当前规则下标签的相对把握」，不是统计显著性 |

## 许可与免责

作者不对使用本工具产生的任何决策后果负责。请勿用于违法用途。
