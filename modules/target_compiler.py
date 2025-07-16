from pathlib import Path
from mindar import MindARCompiler

def compile_targets(images_dir, output_path):
    """Compile targets from images folder"""
    print("🔨 Compiling targets from images folder...")

    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False

    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ No image files found in {images_dir}")
        return False

    try:
        compiler = MindARCompiler()
        success = compiler.compile_directory(images_dir, output_path)
        if success:
            print(f"✅ Targets compiled successfully to {output_path}")
            return True
        else:
            print("❌ Compilation failed")
            return False
    except Exception as e:
        print(f"❌ Failed to compile targets: {e}")
        return False

def load_compiled_targets(targets_file):
    """Load target configuration from compiled targets file"""
    try:
        targets_file = Path(targets_file)
        if targets_file.exists():
            compiler = MindARCompiler()
            mind_data = compiler.load_mind_file(targets_file)
            if mind_data and mind_data.get("dataList"):
                print(f"📋 Loaded {len(mind_data['dataList'])} targets from {targets_file}")
                return mind_data["dataList"]
    except Exception as error:
        print(f"❌ Failed to load target configuration: {error}")
    return []
