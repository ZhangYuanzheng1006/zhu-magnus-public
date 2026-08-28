# zhu-magnus-public — Magnus 集群任务挂载工作区

本仓库是 `zhu-magnus` 工作区的**公开子集**,用途:Magnus 平台任务/蓝图在容器内挂载本仓库作为工作区,
使 runtime 代码(训练/评测/工具脚本)无需打进镜像即可被任务使用。

- 私有仓(完整工作区,含文档/报告/证据):`zhu-magnus`(私有,不公开)
- 本仓库只包含**容器侧需要的 runtime 代码**,由私有仓的 `public/` 子目录通过 `git subtree` 发布同步

## 目录

| 路径 | 内容 |
|---|---|
| `docker/` | 集群环境镜像 Dockerfile(lfm25-env 等) |
| `scripts/` | 容器内执行的 payload 脚本(基准测试、权重下载、冒烟数据、教师推理等) |

## 内容红线(合并前检查)

以下内容**禁止**出现在本仓库:

1. 任何凭据(token/password/api key)或指向私有凭据文件的路径;
2. 内网地址、站点域名、集群内部 job id;
3. 未脱敏的运行日志与个人路径(`C:\Users\...`)。

本地编排/提交器脚本(读取 secret.json 的那部分)留在私有仓,不进本仓库。

## 集群侧使用

```bash
# 容器内(蓝图工作区挂载本仓库后)
python scripts/0828_teacher_smoke.py --model /data/magnus/models/Qwen3.8-27B-20260828
python scripts/0828_make_smoke_data.py --out /data/magnus/smoke-0828/student/data.jsonl
```
