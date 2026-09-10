"""
Project Packager creating a clean StudentAttentionTracker_MVP.zip archive for internship submission.
"""
import os
import zipfile

def create_project_zip(output_zip_path=None):
    """
    Zips the entire StudentAttentionTracker project files into a standalone .zip archive.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if output_zip_path is None:
        output_zip_path = os.path.join(base_dir, 'dist', 'StudentAttentionTracker_MVP.zip')

    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

    ignore_dirs = {'__pycache__', '.git', '.venv', 'venv', 'dist', '.pytest_cache'}
    ignore_extensions = {'.pyc', '.pyo', '.pyd'}

    print(f"Packaging project from {base_dir} to {output_zip_path}...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Exclude ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in ignore_extensions:
                    continue
                
                # Exclude the destination zip file itself if located inside base_dir
                file_path = os.path.join(root, file)
                if os.path.abspath(file_path) == os.path.abspath(output_zip_path):
                    continue

                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)

    size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"Project successfully zipped to {output_zip_path} ({size_mb:.2f} MB)")
    return output_zip_path

if __name__ == '__main__':
    create_project_zip()
