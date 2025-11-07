"""
Fix imports in generated gRPC files after protoc generation.
Run this after generating gRPC code if imports are broken.
"""
from pathlib import Path
import re

def fix_grpc_imports():
    """Fix relative imports in generated gRPC files."""
    grpc_file = Path("generated/vector_service_pb2_grpc.py")
    
    if not grpc_file.exists():
        print(f"❌ File not found: {grpc_file}")
        return False
    
    print(f"Fixing imports in {grpc_file}...")
    content = grpc_file.read_text(encoding='utf-8')
    
    # Pattern 1: import vector_service_pb2 as ...
    pattern1 = r'^import vector_service_pb2 as (.+)$'
    replacement1 = r'from generated import vector_service_pb2 as \1'
    
    content_fixed = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    
    if content != content_fixed:
        grpc_file.write_text(content_fixed, encoding='utf-8')
        print(f"✅ Fixed imports in {grpc_file.name}")
        return True
    else:
        print(f"ℹ️  {grpc_file.name} already has correct imports")
        return True

if __name__ == "__main__":
    success = fix_grpc_imports()
    exit(0 if success else 1)
