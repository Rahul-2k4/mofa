# 复制到 Studio

使用此流程将 Hub 目录条目转换为 MoFA Studio 数据流。

## 片段规范

即用型片段应包含：

- 节点标识符
- 带有安全默认值的必填配置字段
- 最小化的输入/输出连接提示

## 安全注意事项

运行复制的片段前，请务必：

- 确认环境中存在所引用的提供商/模型标识符
- 将占位符密钥替换为本地安全配置
- 检查版本兼容性

## 推荐片段格式

```yaml
name: example-flow
nodes:
  - id: example-node
    type: hub/example-node
    config:
      timeout_ms: 30000
      retries: 1
connections: []
```

保持片段简洁且明确。避免使用隐式默认值，以防跨版本行为发生变化。
