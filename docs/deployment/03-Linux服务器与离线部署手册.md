# Linux服务器与离线部署手册

本文是现有代码的部署操作手册。当前未生成Linux镜像tar，未在甲方服务器验收；Windows部署ZIP不包含Linux Docker镜像，Windows wheelhouse不可用于Linux。

## 1. 条件

Linux x86_64，建议Ubuntu22.04/24.04；Docker Engine、Compose v2（支持 `gpus: all`）、NVIDIA驱动和Container Toolkit；足够可用磁盘。初始建议8核CPU/32GB内存/一张NVIDIA GPU、16GB以上结果空间，具体按数据规模调整。

现有运行时锁使用CUDA13.4库，优先使用兼容该版本的新驱动。当前开发机驱动616.92，服务器驱动版本必须结合具体Linux发行版核查。驱动不是镜像的一部分。

官方参考：

- https://docs.docker.com/engine/install/
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://docs.nvidia.com/deploy/cuda-compatibility/

宿主机安装Container Toolkit后配置Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi
```

重启Docker会影响同机容器，由服务器管理员安排维护窗口。

## 2. 联网构建与启动

将服务器源文件包解压到普通用户可写目录，例如 `/srv/bloodsmear`，包含Dockerfile、Compose、requirements.lock、src、models、samples、deployment、docs。不要上传开发机的.venv、患者样片、data、logs。

```bash
cd /srv/bloodsmear
mkdir -p outputs data logs
export BLOODSMEAR_UID="$(id -u)"
export BLOODSMEAR_GID="$(id -g)"
docker compose build
bash deployment/linux/start.sh
bash deployment/linux/status.sh
bash deployment/linux/smoke-test.sh
```

构建会下载固定Linux依赖。模板、HTMX和中文字体在src中，运行时无公网字体/CDN依赖。只运行一个Uvicorn进程，不自行增加 `--workers`。

浏览器访问 `http://服务器内网IP:8000`；宿主端口由Compose发布，0.0.0.0只是容器内监听地址。

## 3. 数据与权限

现有Compose挂载：

```text
models → /app/models（只读）
outputs → /app/outputs
data → /app/data（SQLite及批量任务）
logs → /app/logs
```

UID/GID取当前启动用户。目录必须归该用户可写，不要用 `chmod 777` 代替正确权限；模型目录需可读。

## 4. 离线镜像包

必须先在与目标同架构的联网Linux构建机成功构建并验证：

```bash
docker compose build
docker save bloodsmear-mvp:0.1.0 -o bloodsmear-mvp-0.1.0.tar
sha256sum bloodsmear-mvp-0.1.0.tar > bloodsmear-mvp-0.1.0.tar.sha256
docker image inspect bloodsmear-mvp:0.1.0 --format '{{.Id}}'
```

交付tar及哈希、源文件包、模型、标签、配置、中文字体许可证、样例、手册。驱动/Toolkit/Container Toolkit如目标机尚无，需管理员另备与OS匹配的离线安装介质；仅镜像tar不能安装这些宿主依赖。

目标断网环境：

```bash
sha256sum -c bloodsmear-mvp-0.1.0.tar.sha256
docker load -i bloodsmear-mvp-0.1.0.tar
bash deployment/linux/start.sh
bash deployment/linux/status.sh
```

启动脚本使用 `--no-build`，防止误触联网构建。验收时真实断网并验证启动、页面、字体、单图和批次报告。

## 5. 停止、更新、恢复

```bash
bash deployment/linux/stop.sh
```

使用 `docker compose stop`保留容器和绑定数据；升级前停止并备份data、outputs、logs。数据库WAL场景先停止服务后整体备份data目录。替换镜像或代码时使用新版本标签，记录哈希并保留旧镜像以便回退。不要删除用户数据目录。

## 6. 服务器验收记录

记录GPU/驱动、OS、Docker/Compose、镜像ID、模型SHA-256、检查日期和负责人。验收：

1. `/health/ready` 返回CUDAExecutionProvider。
2. 单图输出JSON/CSV/标注图/Excel/PDF。
3. independent/grouped分别提交，ZIP有批次和每图报告。
4. 部分坏图不影响成功图输出；零WBC比例为null/“—”。
5. 重启服务恢复批量状态；结果/数据库/日志挂载保留。
6. 日志轮转/保留；Linux符号链接删除保护补测。
7. Python3.11环境复跑P50/P95（包括说明测试口径）；多人请求压力另测。
8. 断网导入镜像、启动、操作通过。

表内“未测试”不得改为“通过”。需要修复服务器差异后才冻结正式镜像与最终验收报告。
