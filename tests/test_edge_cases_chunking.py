import os
from pathlib import Path
from src.processors.markdown_processor import MarkdownProcessor


def test_massive_document():
    """测试超大文档的分块"""
    processor = MarkdownProcessor()
    
    # 创建一个超大的文档（超过100KB）
    massive_content = ""
    
    # 添加大量内容
    for i in range(100):
        massive_content += f"""# 章节 {i}

这是一个非常大的章节，包含大量的内容。

## 子章节 {i}.1

这里有很多内容，用来测试分块算法在处理大文档时的表现。

### 子子章节 {i}.1.1

更多的内容...

### 子子章节 {i}.1.2

更多的内容...

## 子章节 {i}.2

这里也有很多内容...

"""
        
        # 添加大量文本内容
        for j in range(50):
            massive_content += f"这是第{i}章节第{j}段的内容，包含很多文字。" * 10 + "\n\n"
    
    print(f"📏 超大文档测试:")
    print(f"   文档大小: {len(massive_content):,} 字符")
    print(f"   预期分块数: 约 {len(massive_content) // 1000} 个")
    
    # 测试分块
    chunks = processor.split_into_chunks(massive_content, chunk_size=1000, chunk_overlap=200)
    
    print(f"   实际分块数: {len(chunks)}")
    print(f"   平均分块大小: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
    
    # 验证分块质量
    assert len(chunks) > 0, "应该产生分块"
    assert all(len(c) <= 1500 for c in chunks), "所有分块都应该在合理大小范围内"
    
    # 检查是否有分块过大
    oversized_chunks = [c for c in chunks if len(c) > 1200]
    print(f"   过大分块数: {len(oversized_chunks)}")
    
    print("✅ 超大文档测试通过！")


def test_no_headers_document():
    """测试没有标题的文档"""
    processor = MarkdownProcessor()
    
    # 创建一个没有标题的文档
    no_headers_content = """这是一个没有标题的文档。

这里只有普通的段落内容。

更多的内容...

代码示例：
```python
def hello():
    print("Hello World")
```

更多的文本内容...
""" * 100  # 重复100次
    
    print(f"📝 无标题文档测试:")
    print(f"   文档大小: {len(no_headers_content):,} 字符")
    
    # 测试分块
    chunks = processor.split_into_chunks(no_headers_content, chunk_size=500, chunk_overlap=100)
    
    print(f"   分块数: {len(chunks)}")
    print(f"   平均分块大小: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
    
    # 验证分块质量
    assert len(chunks) > 0, "应该产生分块"
    assert all(len(c) <= 750 for c in chunks), "所有分块都应该在合理大小范围内"
    
    # 检查是否回退到简单分块
    chunks_with_headers = [c for c in chunks if '#' in c]
    print(f"   包含标题的分块数: {len(chunks_with_headers)}")
    
    print("✅ 无标题文档测试通过！")


def test_complex_nested_headers():
    """测试复杂的嵌套标题结构"""
    processor = MarkdownProcessor()
    
    # 创建一个有复杂嵌套标题的文档
    complex_content = """# 主标题

## 二级标题1

### 三级标题1.1

#### 四级标题1.1.1

##### 五级标题1.1.1.1

###### 六级标题1.1.1.1.1

内容...

#### 四级标题1.1.2

内容...

### 三级标题1.2

内容...

## 二级标题2

### 三级标题2.1

#### 四级标题2.1.1

内容...

### 三级标题2.2

内容...

# 另一个主标题

## 新的二级标题

内容...
"""
    
    print(f"🏗️  复杂嵌套标题测试:")
    print(f"   文档大小: {len(complex_content):,} 字符")
    
    # 测试分块
    chunks = processor.split_into_chunks(complex_content, chunk_size=200, chunk_overlap=50)
    
    print(f"   分块数: {len(chunks)}")
    
    # 分析每个分块的标题层级
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.split('\n')
        headers = []
        for line in lines:
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                text = line.strip('#').strip()
                headers.append((level, text))
        
        print(f"   分块 {i}:")
        print(f"     大小: {len(chunk)} 字符")
        print(f"     标题层级: {[f'H{h[0]}:{h[1][:20]}' for h in headers]}")
    
    # 验证分块质量
    assert len(chunks) > 0, "应该产生分块"
    
    print("✅ 复杂嵌套标题测试通过！")


def test_mixed_content():
    """测试混合内容（代码块、列表、表格等）"""
    processor = MarkdownProcessor()
    
    # 创建一个包含各种 Markdown 元素的文档
    mixed_content = """# 混合内容测试

## 代码块

```hulo
fn hello() {
    echo "Hello World"
}
```

## 列表

- 项目1
- 项目2
  - 子项目2.1
  - 子项目2.2
- 项目3

## 表格

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |
| 数据4 | 数据5 | 数据6 |

## 引用

> 这是一个引用块
> 包含多行内容

## 强调

**粗体文本** 和 *斜体文本*

## 链接和图片

[链接文本](https://example.com)

![图片描述](image.jpg)

## 更多内容

这里有很多普通文本内容...

""" * 20  # 重复20次
    
    print(f"🔀 混合内容测试:")
    print(f"   文档大小: {len(mixed_content):,} 字符")
    
    # 测试分块
    chunks = processor.split_into_chunks(mixed_content, chunk_size=800, chunk_overlap=150)
    
    print(f"   分块数: {len(chunks)}")
    
    # 检查分块是否保持了 Markdown 结构
    for i, chunk in enumerate(chunks[:3], 1):  # 只检查前3个分块
        print(f"   分块 {i}:")
        print(f"     大小: {len(chunk)} 字符")
        print(f"     包含代码块: {'```' in chunk}")
        print(f"     包含列表: {'-' in chunk or '*' in chunk}")
        print(f"     包含表格: {'|' in chunk}")
        print(f"     包含引用: {'>' in chunk}")
    
    # 验证分块质量
    assert len(chunks) > 0, "应该产生分块"
    
    print("✅ 混合内容测试通过！")


def test_extreme_edge_cases():
    """测试极端边界情况"""
    processor = MarkdownProcessor()
    
    # 测试空文档
    empty_chunks = processor.split_into_chunks("")
    print(f"📭 空文档测试: {len(empty_chunks)} 个分块")
    assert len(empty_chunks) == 0 or len(empty_chunks[0]) == 0, "空文档应该产生空分块或0个分块"
    
    # 测试只有标题的文档
    headers_only = "# 标题1\n## 标题2\n### 标题3"
    header_chunks = processor.split_into_chunks(headers_only)
    print(f"📋 只有标题的文档: {len(header_chunks)} 个分块")
    assert len(header_chunks) > 0, "应该产生分块"
    
    # 测试超大标题
    huge_header = "# " + "非常大的标题" * 1000
    huge_chunks = processor.split_into_chunks(huge_header)
    print(f"📏 超大标题: {len(huge_chunks)} 个分块")
    assert len(huge_chunks) > 0, "应该产生分块"
    
    # 测试特殊字符
    special_chars = "# 特殊字符测试\n\n包含 `代码` 和 **粗体** 和 [链接](url) 的内容"
    special_chunks = processor.split_into_chunks(special_chars)
    print(f"🔤 特殊字符: {len(special_chunks)} 个分块")
    assert len(special_chunks) > 0, "应该产生分块"
    
    print("✅ 极端边界情况测试通过！")


if __name__ == "__main__":
    print("🚀 开始边界情况压力测试...\n")
    
    test_massive_document()
    print()
    
    test_no_headers_document()
    print()
    
    test_complex_nested_headers()
    print()
    
    test_mixed_content()
    print()
    
    test_extreme_edge_cases()
    print()
    
    print("🎉 所有边界情况测试完成！") 