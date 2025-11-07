"""
Script to generate Python gRPC code from .proto files.
Run: python generate_grpc.py
"""
import subprocess
import sys
from pathlib import Path
import re


def patch_imports(file_path):
    """
    Patch the generated gRPC files to use absolute imports.
    Converts: import vector_service_pb2
    To: from generated import vector_service_pb2
    """
    content = file_path.read_text(encoding='utf-8')
    
    # Pattern to match: import vector_service_pb2
    pattern = r'^import vector_service_pb2 as (.+)$'
    replacement = r'from generated import vector_service_pb2 as \1'
    
    content_patched = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != content_patched:
        file_path.write_text(content_patched, encoding='utf-8')
        return True
    return False


def generate_grpc_code():
    proto_dir = Path("protos")
    output_dir = Path("generated")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Generate Python code
    proto_file = proto_dir / "vector_service.proto"
    
    if not proto_file.exists():
        print(f"❌ Error: {proto_file} not found!")
        sys.exit(1)
    
    print(f"Generating gRPC code from {proto_file}...")
    
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Generated files in {output_dir}/")
        print(f"   - vector_service_pb2.py")
        print(f"   - vector_service_pb2_grpc.py")
        
        # NEW: Patch imports in generated files
        print("\n🔧 Patching imports for absolute references...")
        grpc_file = output_dir / "vector_service_pb2_grpc.py"
        
        if grpc_file.exists():
            if patch_imports(grpc_file):
                print(f"   ✓ Patched {grpc_file.name}")
            else:
                print(f"   ℹ  {grpc_file.name} already uses correct imports")
        
    else:
        print(f"❌ Error generating gRPC code:")
        print(result.stderr)
        sys.exit(1)
    
    # Create __init__.py
    init_file = output_dir / "__init__.py"
    init_file.write_text("# Generated gRPC code\n", encoding='utf-8')
    
    print("\n✅ gRPC code generation complete!")
    print("You can now import: from generated import vector_service_pb2, vector_service_pb2_grpc")


if __name__ == "__main__":
    generate_grpc_code()
