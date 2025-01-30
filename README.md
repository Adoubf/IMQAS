# 智能医疗问答系统
## 项目背景

本项目基于 WeRoBot、Flask、BERT、Redis、Neo4j 等技术，构建了一个完整的 微信公众号智能问答系统，支持 文本分析、知识库构建 和 智能检索。该系统主要包含 微信公众号管理、自然语言处理（NLP）分析、缓存管理、知识图谱 等核心模块。

## 目录结构

```
.
├── docker_offline         # 离线数据处理
│   ├── dataset            # 数据集存储
│   ├── model              # 预训练模型
│   ├── output             # 处理后数据存储
│   ├── preprocessing_structured_data   # 结构化数据预处理
│   ├── preprocessing_unstructured_data # 非结构化数据预处理
├── docker_online          # 在线服务模块
│   ├── bert_serve         # BERT 服务
│   ├── main_serve         # 主服务端
│   ├── wechat             # 微信公众号管理
├── logs                   # 全局日志存储
├── wr.py                  
```

## 1. 运行环境

### 1.1 依赖安装

本项目依赖以下 Python 库及工具：

```bash
WeRoBot~=1.13.1
requests~=2.32.3
zhipuai~=2.1.5.20250106
torch~=2.5.1
flask~=3.1.0
pandas~=2.2.3
scikit-learn~=1.6.1
transformers~=4.45.2
redis~=5.2.1
neo4j~=5.27.0
matplotlib~=3.9.2
tqdm~=4.66.5
numpy~=1.26.4
datasets~=2.19.1
torchvision~=0.20.1
pydantic~=2.10.6
```

### 1.2 服务器环境

- Debian GNU/Linux 11
- 4H8G
- 无GPU

## 2. 各模块介绍

### 2.1 微信公众号管理

- **WeRoBot**：用于处理微信公众号的消息交互。
- **Flask Web 服务器**：提供 REST API，连接微信、NLP 模块。
- **智普智能对话**：调用智普 API 进行对话处理。
- **Redis 缓存**：加速访问，存储临时数据。
- **Neo4j 知识图谱**：存储结构化的医疗知识，支持知识查询。

### 2.2 语义处理核心模块（NLP）

- **BERT 语义匹配**：对文本进行语义理解，提高问答匹配精度。
- **RNN 语义分析**：结合历史数据，进行深度语义学习。
- **命名实体识别（NER）**：提取关键实体信息（如疾病、症状）。

### 2.3 数据存储

- **结构化数据**：存储于 Neo4j，便于查询。
- **非结构化数据**：通过 NER、BERT 进行文本解析。
- **Redis**：存储高频查询结果，优化性能。
## 3. 部署与启动方式

本项目统一使用 **Supervisord** 进行进程管理，并使用 **Gunicorn** 作为 Web 服务器。

### 3.1 Supervisord 配置

#### 3.1.1 启动微信公众号后端

```
[program:wechat_bot]
command=/root/anaconda3/envs/请替换为自己的真实虚拟环境路径/bin/python 请替换为自己的真实项目路径/wr.py
autostart=true
autorestart=true
startsecs=5
stderr_logfile=请替换为自己的真实项目路径/logs/wechat_bot.err.log
stdout_logfile=请替换为自己的真实项目路径/logs/wechat_bot.out.log
stderr_logfile_maxbytes=50MB
stdout_logfile_maxbytes=50MB
stderr_logfile_backups=5
stdout_logfile_backups=5
user=root
directory=请替换为自己的真实项目路径
environment=PYTHONUNBUFFERED=1
```

#### 3.1.2 启动 Flask 问答接口

```
[program:reply_end]
command=/root/anaconda3/envs/请替换为自己的真实虚拟环境路径/bin/gunicorn -w 1 -b 0.0.0.0:5000 app:app
autostart=true
autorestart=true
startsecs=10
stderr_logfile=请替换为自己的真实项目路径/logs/reply_end.err.log
stdout_logfile=请替换为自己的真实项目路径/logs/reply_end.out.log
stderr_logfile_maxbytes=50MB
stdout_logfile_maxbytes=50MB
stderr_logfile_backups=5
stdout_logfile_backups=5
user=root
directory=请替换为自己的真实项目路径/docker_online/main_serve
environment=PYTHONUNBUFFERED=1
```

#### 3.1.3 启动 BERT 识别服务

```
[program:recognition_end]
command=/root/anaconda3/envs/请替换为自己的真实虚拟环境路径/bin/gunicorn -w 1 -b 0.0.0.0:5001 app:app
autostart=true
autorestart=true
startsecs=10
stderr_logfile=请替换为自己的真实项目路径/logs/recognition_end.err.log
stdout_logfile=请替换为自己的真实项目路径/logs/recognition_end.out.log
stderr_logfile_maxbytes=50MB
stdout_logfile_maxbytes=50MB
stderr_logfile_backups=5
stdout_logfile_backups=5
user=root
directory=请替换为自己的真实项目路径/docker_online/bert_serve
environment=PYTHONUNBUFFERED=1
```

## 4. API 说明

| API 地址            | 描述               |
| ------------------- | ------------------ |
| `/wr.py`            | 微信公众号消息处理 |
| `/zhipu_ai_chat.py` | 智普AI 对话        |
| `/v1/recognition/`  | BERT 语义匹配      |
| `/v1/main_serve/`   | 命名实体识别       |

## 5. 贡献指南

如需贡献代码，请遵循以下流程：

1. Fork 本项目
2. 创建新分支：`git checkout -b feature-new`
3. 提交代码：`git commit -m '新增功能'`
4. Push 到远程仓库：`git push origin feature-new`
5. 提交 PR（Pull Request）

## 6. 许可证

本项目基于 MIT 许可证发布，可自由使用和修改。

------

如有问题，请联系 haoyue@svipdagr.com。
