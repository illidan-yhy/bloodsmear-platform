# 外周血涂片图像分析系统

本系统提供单图和批量细胞识别、数量与比例统计、标注图以及 Excel/PDF 报告，供科研与内部分析使用。

## 安装与启动

1. 将部署ZIP完整解压到当前用户可写目录。
2. 根据 `部署手册.html` 安装 Python 3.11 x64、VC++ x64运行库和兼容的NVIDIA显卡驱动。
3. 双击 `04_collect_info.cmd` 采集电脑配置。
4. 双击 `01_install.cmd` 联网安装依赖并检查GPU。
5. 双击 `02_start.cmd` 启动服务，浏览器打开 `http://127.0.0.1:8000`。

首次安装需要联网下载依赖，完成后日常运行无需联网。保持启动窗口打开，在窗口按 `Ctrl+C` 可正常停止。

## 输入与模式

- 支持普通 JPG、PNG、TIF 图片。
- independent：每张图片作为独立样本。
- grouped：多张图片属于同一样本的多个视野，需要填写共同的样本ID。
- 每批最多100张，单张最多25 MiB，整批最多500 MiB。

识别类别包括 Basophil、Eosinophil、Lymphocyte、Monocyte、Neutrophil、Platelets 和 RBC。

## 结果与报告

单图可下载JSON、CSV、标注图、Excel和PDF。批量完成后下载ZIP，其中包含批次汇总报告以及每张成功图片的独立Excel/PDF和基础结果。PDF只包含文本与表格。

WBC比例以五类白细胞总数为分母，不包含RBC和Platelets。未检测到白细胞时比例显示“—”，质控提示 `NO_WBC_DETECTED`。

## 数据与日志

- 单图结果位于 `outputs/`。
- 批量任务与数据库位于 `data/`。
- 日志位于 `logs/app.log`，每日轮转，默认保留14天。
- 批量原图默认保留24小时，批量结果默认保留7天。需要长期保存时及时下载。

## 配置与检查

安装后可通过 `.env` 调整端口、日志及保留配置。默认仅本机访问。`03_status.cmd` 检查服务状态，GPU正常运行时Provider为 `CUDAExecutionProvider`。

`outputs/computer-info.txt` 和 `outputs/environment-check.json` 用于确认目标电脑配置及部署结果。

## 完整说明

直接打开 `部署手册.html`，或查看 `docs/deployment/` 中的Windows部署手册、环境信息采集表和交付包清单。

原始包应保持文件完整；如需编辑交付文档，请先修改源文件并重新生成ZIP和校验清单，避免安装时出现校验不匹配。
