"""Check the curated publication boundary; this complements human review."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.py', '.md', '.json', '.html', '.png', '.txt', '.svg', '.toml', '.yml', '.yaml'}
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'outputs', 'artifacts'}
failures = []
files = []
for path in sorted(ROOT.rglob('*')):
    rel = path.relative_to(ROOT)
    if any(p in SKIP for p in rel.parts):
        continue
    if path.is_symlink():
        failures.append(f'{rel}: symlink')
        continue
    if not path.is_file():
        continue
    files.append(str(rel))
    if path.suffix not in ALLOWED and path.name not in {'LICENSE', '.gitignore'}:
        failures.append(f'{rel}: unexpected file type')
    if path.suffix == '.png':
        continue
    text = path.read_text()
    # Never output matching content: source locations alone are sufficient.
    for label, regex in (
        ('private filesystem path', r'/(?:Users|home)/[A-Za-z0-9_-]+/'),
        ('private key material', r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'),
        ('service credential', r'"type"\s*:\s*"service_account"'),
        ('Google API key', r'AIza[0-9A-Za-z_-]{30,}'),
        ('GitHub token', r'gh[pousr]_[A-Za-z0-9]{30,}'),
    ):
        if re.search(regex, text):
            failures.append(f'{rel}: {label}')
    if path.suffix == '.md':
        for target in re.findall(r'\]\(([^\s)]+)\)', text):
            if re.match(r'^[a-zA-Z]+:', target) or target.startswith('#'):
                continue
            if not (path.parent / target.split('#')[0]).exists():
                failures.append(f'{rel}: missing local link {target}')
verified_hashes = 0
for manifest in (ROOT/'SOURCE_MANIFEST.json', ROOT/'twin/SOURCE_MANIFEST.json'):
    content = json.loads(manifest.read_text())
    entries = content if isinstance(content, list) else content['files']
    for entry in entries:
        snapshot = entry.get('snapshot_path', entry.get('snapshot'))
        expected = entry.get('snapshot_sha256', entry.get('sha256'))
        if hashlib.sha256((ROOT/snapshot).read_bytes()).hexdigest() != expected:
            failures.append(f'{snapshot}: source manifest hash mismatch')
        verified_hashes += 1
example = json.loads((ROOT/'examples/saved-synthetic-brief.json').read_text())
assert example['syntheticOnly'] is True and example['savedExampleSubset'] is True
for history in example['histories']:
    assert history['brief']['synthetic'] is True
    for item in history['brief']['items']:
        assert item['evidenceId'] in history['brief']['evidence']
html = (ROOT/'examples/saved-synthetic-brief.html').read_text()
assert not re.search(r'<(?:script|iframe)\b|(?:src|href)=["\']https?://', html, re.I)
print(json.dumps({'files_checked':len(files), 'source_hashes_checked':verified_hashes, 'failures':failures}, indent=2))
raise SystemExit(bool(failures))
