"""Assemble paper/CAPTIONS.md from the fragments in paper/captions (written by the figN.py scripts)."""
from common import *
parts = sorted(CAPD.glob('*.md'), key=lambda p: (p.stem[0] != 'f', p.stem.startswith('S'), p.stem))
out = ['# Caption drafts (numbers computed from paper/data/*.csv by the figure scripts; do not edit by hand)\n']
for p in parts: out.append(f'## {p.stem}\n\n' + p.read_text().strip() + '\n')
(PAPER/'CAPTIONS.md').write_text('\n'.join(out)); print('wrote', PAPER/'CAPTIONS.md', len(parts), 'fragments')
