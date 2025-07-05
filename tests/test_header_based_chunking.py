import os
from pathlib import Path
from src.processors.markdown_processor import MarkdownProcessor


def test_header_based_chunking():
    """测试基于标题的分块策略"""
    processor = MarkdownProcessor()
    
    # 创建一个测试用的 Markdown 内容
    test_content = """# 主标题

这是主标题下的内容。

## 二级标题1

这是二级标题1的内容。

### 三级标题1.1

这是三级标题1.1的内容。

### 三级标题1.2

这是三级标题1.2的内容。

## 二级标题2

这是二级标题2的内容。

### 三级标题2.1

这是三级标题2.1的内容。

# 另一个主标题

这是另一个主标题的内容。
"""
    
    # 测试分块
    chunks = processor.split_into_chunks(test_content, chunk_size=200, chunk_overlap=50)
    
    print("📝 基于标题的分块测试:")
    print(f"   原始内容长度: {len(test_content)} 字符")
    print(f"   分块数量: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   分块 {i}:")
        print(f"     长度: {len(chunk)} 字符")
        print(f"     内容预览: {chunk[:100]}...")
        print(f"     是否包含标题: {'是' if '#' in chunk else '否'}")
        print()
    
    # 验证分块质量
    assert len(chunks) > 0, "应该产生分块"
    
    # 检查每个分块是否包含标题
    chunks_with_headers = [c for c in chunks if '#' in c]
    print(f"   包含标题的分块数: {len(chunks_with_headers)}/{len(chunks)}")
    
    # 检查分块大小是否合理
    chunk_sizes = [len(c) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    print(f"   平均分块大小: {avg_size:.0f} 字符")
    
    assert avg_size <= 300, "平均分块大小应该在合理范围内"
    
    print("✅ 基于标题的分块测试通过！")


def test_real_document_chunking():
    """测试真实文档的分块效果"""
    processor = MarkdownProcessor()
    
    # 读取一个真实的文档
    docs_path = Path("../docs/src/grammar/stmt.md")
    if not docs_path.exists():
        print("⚠️  真实文档不存在，跳过测试")
        return
    
    with open(docs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 测试分块
    chunks = processor.split_into_chunks(content)
    
    print("📄 真实文档分块测试:")
    print(f"   文档: {docs_path.name}")
    print(f"   原始长度: {len(content)} 字符")
    print(f"   分块数量: {len(chunks)}")
    print()
    
    # 分析分块
    header_chunks = 0
    total_headers = 0
    
    for i, chunk in enumerate(chunks[:5], 1):  # 只显示前5个分块
        headers_in_chunk = chunk.count('#')
        total_headers += headers_in_chunk
        
        if headers_in_chunk > 0:
            header_chunks += 1
        
        print(f"   分块 {i}:")
        print(f"     长度: {len(chunk)} 字符")
        print(f"     标题数量: {headers_in_chunk}")
        
        # 提取第一个标题
        lines = chunk.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                print(f"     第一个标题: {line.strip()}")
                break
        print()
    
    print(f"   包含标题的分块: {header_chunks}/{len(chunks)}")
    print(f"   总标题数: {total_headers}")
    
    # 验证分块质量
    assert len(chunks) > 0, "应该产生分块"
    assert header_chunks > 0, "应该有包含标题的分块"
    
    print("✅ 真实文档分块测试通过！")


if __name__ == "__main__":
    test_header_based_chunking()
    test_real_document_chunking() 