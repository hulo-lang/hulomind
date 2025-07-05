from src.processors.markdown_processor import MarkdownProcessor

def debug_chunking():
    processor = MarkdownProcessor()
    
    # 创建一个测试文档
    test_content = """# 测试文档

## 章节1

这是章节1的内容。

### 子章节1.1

这是子章节1.1的内容。

### 子章节1.2

这是子章节1.2的内容。

## 章节2

这是章节2的内容。

### 子章节2.1

这是子章节2.1的内容。
"""
    
    print("测试文档内容:")
    print(test_content)
    print(f"文档大小: {len(test_content)} 字符")
    print()
    
    # 测试分块
    chunks = processor.split_into_chunks(test_content, chunk_size=100, chunk_overlap=20)
    
    print(f"分块结果:")
    print(f"分块数量: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"分块 {i}:")
        print(f"  大小: {len(chunk)} 字符")
        print(f"  内容: {repr(chunk[:100])}...")
        print()

if __name__ == "__main__":
    debug_chunking() 