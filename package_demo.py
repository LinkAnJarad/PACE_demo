import os
import sys
import time
import zipfile

def package_demo():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    zip_path = os.path.join(parent_dir, "predictive_processing_demo.zip")

    # Exclusions to prevent bloat and cross-platform issues
    EXCLUDE_DIRS = {
        ".venv",
        "venv",
        "env",
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "dist-ssr",
        ".idea",
        ".vscode",
    }
    EXCLUDE_FILES = {
        ".env",
        ".env.local",
        "desktop.ini",
        "Thumbs.db",
        ".DS_Store",
    }
    EXCLUDE_EXTS = {".pyc", ".pyo", ".tmp"}

    print("=" * 70)
    print("  Predictive Processing Demo - Standalone Packager")
    print("=" * 70)
    print(f"Source Folder: {script_dir}")
    print(f"Output Zip:    {zip_path}")
    print()
    print("Collecting files to bundle (excluding .venv, .git, temp files)...")

    files_to_pack = []
    total_bytes = 0

    for root, dirs, files in os.walk(script_dir):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file in EXCLUDE_FILES:
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in EXCLUDE_EXTS:
                continue
            
            full_path = os.path.join(root, file)
            # Relative path inside the zip archive, prefixed by folder name
            rel_path = os.path.relpath(full_path, parent_dir)
            
            try:
                size = os.path.getsize(full_path)
                total_bytes += size
                files_to_pack.append((full_path, rel_path, size))
            except Exception as e:
                print(f"  Warning: Skipping {file} ({e})")

    total_mb = total_bytes / (1024 * 1024)
    print(f"Found {len(files_to_pack)} files ({total_mb:.2f} MB).")
    print("Creating compressed ZIP archive (this typically takes 10-20 seconds)...")

    start_time = time.time()
    packed_bytes = 0

    # If old zip exists, remove it
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception:
            pass

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for i, (full_path, rel_path, size) in enumerate(files_to_pack, 1):
            zipf.write(full_path, arcname=rel_path)
            packed_bytes += size
            pct = (packed_bytes / total_bytes) * 100 if total_bytes > 0 else 100
            if i % 25 == 0 or i == len(files_to_pack):
                sys.stdout.write(f"\r  Compressing: [{i}/{len(files_to_pack)}] {pct:5.1f}%  ({os.path.basename(full_path)[:30]:<30})")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)

    print()
    print()
    print("=" * 70)
    print("  [SUCCESS] Demo packaged successfully!")
    print(f"  Archive:       {zip_path}")
    print(f"  Archive Size:  {zip_size_mb:.2f} MB  (compressed from {total_mb:.2f} MB)")
    print(f"  Elapsed Time:  {elapsed:.1f} seconds")
    print(f"  Files Bundled: {len(files_to_pack)} clean files")
    print("=" * 70)
    print("  Ready to send to your supervisor via Google Drive, AirDrop, or USB.")
    print("  Your supervisor can extract it on macOS and run `./start_demo.sh` directly.")
    print("=" * 70)

if __name__ == "__main__":
    package_demo()
