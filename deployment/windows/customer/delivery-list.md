# Windows交付包清单与验收记录

日期：2026-10-08。软件版本：0.1.0。

## 包内文件

- 应用代码与本地网页静态资源。
- ONNX模型、类别映射、推理配置和模型哈希。
- 中文报告字体及字体许可证。
- `samples/Blood.png` 示例图片。
- `requirements.lock` 固定运行依赖清单。
- 安装、启动、状态检查、停止、环境采集及包校验脚本。
- Windows部署手册、环境采集表、交付包清单和浏览器版手册。
- `PACKAGE_MANIFEST.json` 逐文件SHA-256校验清单。

联网版不含Python、VC++、NVIDIA驱动安装程序或Python离线依赖。首次安装需要联网，安装完成后系统可本地运行。

## 目标电脑验收

目标电脑的部署状态以实际安装和检查结果为准。安装后填写以下记录：

| 项目 | 验收方法 | 记录 |
|---|---|---|
| 操作系统与GPU | `04_collect_info.cmd` | 待检查 |
| Python与运行依赖 | `01_install.cmd` | 待检查 |
| 实际GPU推理 | environment-check.json内passed=true，Provider为CUDAExecutionProvider | 待检查 |
| 单图结果与Excel/PDF | 网页上传样例并下载报告 | 待检查 |
| 两种批量模式 | 创建independent/grouped任务 | 待检查 |
| 批次及每图报告 | 解压结果ZIP检查报告 | 待检查 |
| 服务启停与日志 | 启停脚本及logs/app.log | 待检查 |

检查日期、负责人及结果可另存为验收记录。已经打包的原始文件不应直接编辑；需调整交付材料时重新打包生成匹配的校验清单。
