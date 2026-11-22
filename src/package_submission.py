"""Package project deliverables and selected weights without data or environments."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    # Explicit allowlist prevents accidentally shipping environments or secrets.
    files = [ROOT / name for name in [
        'README.md', '.gitignore', '第二次学习过程记录.md', '提交说明.md',
        'report/experiment_report.md', 'report/experiment_report.tex',
        'report/experiment_report.pdf',
    ]]
    for directory, suffixes in [
        ('src', {'.py'}), ('configs', {'.yaml', '.json', '.csv'}),
        ('outputs', {'.json', '.csv', '.yaml', '.png'}),
        ('.agents/skills/pytorch-image-classification', {'.md', '.yaml'}),
    ]:
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in suffixes
                     and '__pycache__' not in p.parts)
    for experiment in ('mlp', 'cnn', 'cnn-flip'):
        files.append(ROOT / 'checkpoints' / experiment / 'best.pt')
        for name in ('history.csv', 'metrics.json', 'predictions.csv',
                     'confusion-matrix.png', 'examples.png', 'run_metadata.json'):
            if not (ROOT / 'outputs' / experiment / name).is_file():
                raise FileNotFoundError(f'Missing artifact: {experiment}/{name}')
    files = sorted(set(files))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        path.resolve().relative_to(ROOT.resolve())

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_commit': commit, 'working_tree_dirty': dirty, 'files': []}
    destination = ROOT.parent / '提交资料'
    destination.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    archive = destination / f'imagenette-classification-submission-{stamp}.zip'
    prefix = 'imagenette-classification/'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            manifest['files'].append({'path': name, 'bytes': len(data), 'sha256': digest(data)})
            bundle.writestr(prefix + name, data)
        bundle.writestr(prefix + 'submission-manifest.json',
                        json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed')
        for entry in manifest['files']:
            if digest(bundle.read(prefix + entry['path'])) != entry['sha256']:
                raise RuntimeError(f"Hash verification failed: {entry['path']}")
        assert len(bundle.namelist()) == len(files) + 1
    archive_hash = digest(archive.read_bytes())
    archive.with_suffix('.zip.sha256').write_text(
        f'{archive_hash}  {archive.name}\n', encoding='utf-8')
    print(f'Archive: {archive}')
    print(f'Files: {len(files)} + manifest; all CRC and SHA256 checks passed')
    print(f'Size: {archive.stat().st_size / 1024 / 1024:.2f} MiB')
    print(f'Source commit: {commit}; working tree dirty: {dirty}')
    print(f'SHA256: {archive_hash}')


if __name__ == '__main__':
    main()
