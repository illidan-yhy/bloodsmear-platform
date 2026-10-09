# Windows RTX 3060 试用部署手册

适用版本：BloodSmear 0.1.0 Windows试用版，2026-10-08。

已知甲方条件：Windows、显卡约 RTX 3060、安装时可以联网。Windows版本、显存、内存和驱动尚未确认。本文提供检查步骤，不能替代甲方电脑上的实测结论。

## 1. 先确认电脑条件

建议 Windows 10/11 **64位**，RTX 3060，内存16GB及以上，SSD可用空间至少15GB（依赖下载、缓存与虚拟环境会占用数GB；图片与结果另计）。有浏览器即可，不需要安装 Microsoft Office 才能生成 Excel。

目前固定依赖为 Python 3.11 x64、ONNX Runtime GPU 1.30.0、CUDA 13.4 系列 Python运行库、cuDNN 9。推荐 NVIDIA 驱动 **616.92或更新的兼容版本**；旧驱动不能只凭显卡型号判定兼容。安装适用于 GeForce RTX 3060 的官方驱动，不使用数据中心显卡驱动。

不需要单独安装完整CUDA Toolkit、PyTorch、TensorRT、Docker或Node.js。CUDA/cuDNN运行库由 Python依赖安装。驱动是宿主机依赖，仍需自行安装。

官方说明：

- ONNX Runtime GPU/CUDA版本匹配：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- NVIDIA驱动下载：https://www.nvidia.com/Download/index.aspx
- CUDA驱动兼容说明：https://docs.nvidia.com/deploy/cuda-compatibility/
- Windows VC++ x64运行库：https://aka.ms/vs/17/release/vc_redist.x64.exe
- Python 3.11.9 Windows x64安装程序：https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

Python 3.11.9用于未安装Python的Windows电脑；已安装3.11.x x64可以复用。安装后须执行环境检查。

## 2. 解压与安装

将Windows部署ZIP完整解压到当前用户可写目录，例如 `D:\BloodSmear`。不要在ZIP内直接运行，不要复制其他电脑的 `.venv`。避免放到需要管理员写权限的 `Program Files`。

包内包含代码、模型、中文字体、标签/阈值配置、样例、依赖锁、操作脚本和手册。联网版**不包含Python安装程序、NVIDIA驱动安装程序和Python wheel离线依赖**。

1. 安装Python 3.11 x64（已有可跳过）；可勾选Python launcher。
2. 首次使用建议安装/修复Microsoft VC++ 2015–2022 x64运行库。
3. 更新NVIDIA驱动后，按系统提示重启。
4. 双击 `04_collect_info.cmd`，产生 `outputs/computer-info.txt`。
5. 双击 `01_install.cmd`，联网安装固定依赖并执行真实GPU推理与报告检查。

如电脑有多个Python且自动发现失败，在PowerShell执行：

```powershell
cd D:\BloodSmear
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -PythonExe 'C:\实际Python311路径\python.exe'
```

安装成功后应显示 `Installation and GPU smoke test passed`。环境证据保存在 `outputs/environment-check.json`，实际Provider必须是 `CUDAExecutionProvider`，`passed`应为true。只出现可用Provider列表不等于模型已经在GPU运行。

## 3. 启动与访问

双击 `02_start.cmd`，等待出现Uvicorn启动信息，保持窗口打开。浏览器访问：

```text
http://127.0.0.1:8000
```

无需VPN。默认 `.env` 绑定 `127.0.0.1`，仅本机访问。`0.0.0.0`是监听地址，不是浏览器访问地址。

双击 `03_status.cmd` 检查就绪状态。也可访问：

```text
http://127.0.0.1:8000/health/ready
```

更换端口时编辑 `.env` 中 `BLOODSMEAR_PORT`，状态脚本用 `03_status.cmd -Port 新端口`。

## 4. 效果验收

先上传 `samples/Blood.png`。它是发布者提供的带标注拼图，仅用于软件冒烟测试；不适合据此判断模型准确率。当前固定阈值下基线为55个目标（52 RBC、2 Monocyte、1 Platelets）。

- 单图：确认标注图、JSON、CSV、Excel、PDF下载可用。
- independent：选择两张图，每图按独立样本处理。
- grouped：选择两个视野，填写同一个样本ID，汇总数量后重新计算比例。
- 下载批量ZIP：根目录有整体Excel/PDF；`items/<item_id>/`有每图Excel/PDF及基础结果。
- 再用甲方自有清晰图像查看效果；模型识别质量由算法侧负责。

WBC表示五类白细胞之和，Monocyte只是其中之一。未检出WBC时，比例显示“—”，QC提示 `NO_WBC_DETECTED`，属于不可计算提示。

## 5. 停止、数据和日志

推荐在启动窗口按 `Ctrl+C` 正常停止。`05_stop.cmd`只匹配当前目录的Python服务进程，会强制终止；未完成批次下次启动会恢复。

- 单图产物：`outputs/`。
- 批量数据库与任务：`data/`。
- 日志：`logs/app.log`，按宿主机本地时间每日轮转，默认保留14天。
- 批量原图默认24小时、结果7天，终态任务到期会清理；单图 `outputs/`当前没有相同的自动保留清理机制。

升级时先停止服务并备份 `.env`、`data`、`outputs`、`logs`。每个部署目录启动一个服务进程。

## 6. 常见问题

| 现象 | 操作 |
|---|---|
| 找不到Python | 指定3.11 x64解释器路径；不使用3.12/Anaconda替代本包环境 |
| DLL加载失败或CUDA不可用 | 检查NVIDIA驱动、VC++ x64运行库，重新执行安装检查；不要同时装CPU版onnxruntime |
| CPU_FALLBACK | GPU检查未通过，不能按GPU性能验收；修复驱动/运行库后重启 |
| pip下载失败 | 查看安装窗口错误；甲方可联网时重试；仅在组织批准下使用镜像源 |
| 8000被占用 | 停止旧服务或改 `.env` 端口；不结束其他程序 |
| 浏览器打不开 | 使用127.0.0.1，检查服务窗口和状态脚本；本机浏览器代理须绕过回环地址 |
| 路径拒绝访问 | 移至用户可写目录；确认安全软件没有阻止Python写入数据目录 |

## 7. 验证边界

目标电脑应完成安装检查并产生 `environment-check.json`，其中passed为true、实际Provider为CUDAExecutionProvider，才可记录该电脑GPU部署通过。

模型README与内置许可证标记冲突尚未关闭，状态为 `pending_publisher_confirmation`；交付商业授权由项目负责人确认。报告保留科研用途声明。
