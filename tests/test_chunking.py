import os
from pathlib import Path
from src.processors.document_loader import DocumentLoader
from src.config.settings import settings


def test_chunking_effectiveness():
    """测试分块效果"""
    loader = DocumentLoader()
    
    # 加载文档
    docs = loader.load_documents(max_workers=2)
    assert len(docs) > 0, "应该能加载到文档"
    
    # 创建分块
    chunks = loader.create_all_chunks(docs)
    assert len(chunks) > 0, "应该能创建分块"
    
    print(f"📊 分块统计:")
    print(f"   原始文档数: {len(docs)}")
    print(f"   总分块数: {len(chunks)}")
    print(f"   平均每文档分块数: {len(chunks) / len(docs):.1f}")
    
    # 检查分块大小
    chunk_sizes = [len(chunk.content) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    max_size = max(chunk_sizes)
    min_size = min(chunk_sizes)
    
    print(f"   分块大小统计:")
    print(f"     平均: {avg_size:.0f} 字符")
    print(f"     最大: {max_size} 字符")
    print(f"     最小: {min_size} 字符")
    print(f"     配置的chunk_size: {settings.chunk_size}")
    
    # 验证分块质量
    assert avg_size <= settings.chunk_size * 1.2, "平均分块大小应该在合理范围内"
    assert max_size <= settings.chunk_size * 1.5, "最大分块不应该过大"
    
    # 检查语言分布
    en_chunks = [c for c in chunks if c.language == "en"]
    zh_chunks = [c for c in chunks if c.language == "zh"]
    
    print(f"   语言分布:")
    print(f"     英文分块: {len(en_chunks)}")
    print(f"     中文分块: {len(zh_chunks)}")
    
    # 检查分类分布
    categories = {}
    for chunk in chunks:
        cat = chunk.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   分类分布:")
    for cat, count in sorted(categories.items()):
        print(f"     {cat}: {count} 个分块")
    
    print("✅ 分块测试通过！")


def test_chunk_content_quality():
    """测试分块内容质量"""
    loader = DocumentLoader()
    docs = loader.load_documents(max_workers=2)
    chunks = loader.create_all_chunks(docs)
    
    # 检查分块内容质量
    for chunk in chunks[:5]:  # 检查前5个分块
        assert len(chunk.content.strip()) > 0, "分块内容不应为空"
        assert not chunk.content.startswith("---"), "分块不应以frontmatter开始"
        assert chunk.chunk_index >= 0, "分块索引应该有效"
        assert chunk.start_pos < chunk.end_pos, "分块位置应该有效"
    
    print("✅ 分块内容质量测试通过！")


if __name__ == "__main__":
    test_chunking_effectiveness()
    test_chunk_content_quality() 