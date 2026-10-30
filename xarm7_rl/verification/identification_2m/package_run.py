"""Preserve 2 m training attempts, selection, all controls, and actual recordings."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import runpy
import sys
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='runs/xarm7_identification_2m')
    parser.add_argument('--output',default='xarm7_rl/verification/identification_2m')
    args=parser.parse_args();source,root=Path(args.run),Path(args.output)
    helpers=runpy.run_path(str(Path(__file__).parent.parent/'identification/package_run.py'))
    copy,evaluations=helpers['copy'],helpers['evaluations']
    copy(source/'protocol.json',root/'protocol.json')
    protocol=json.loads((source/'protocol.json').read_text())
    for case in ['center_060','center_100','corner_100']:
        copy(source/protocol['selected_from'][case]/'selected.pt',root/'checkpoints'/f'{case}.pt')
        copy(source/'calibrated_initial'/f'{case}.pt',root/'initialization'/f'{case}.pt')
        for prefix in ['', 'warm_', 'calibrated_']:
            name=prefix+case
            for file in ['config.yaml','metrics.jsonl']:
                copy(source/name/file,root/'training'/name/file)
            copy(source/f'{name}.log',root/'training'/name/'console.log')
        copy(source/f'candidates_{case}'/'manifest.json',root/'validation'/f'candidates_{case}.json')
    for selection in sorted(source.glob('selection_*')):
        if selection.is_dir():evaluations(selection,root/'validation'/selection.name)
    for group in ['baseline','pilot','calibrated_initial']:
        evaluations(source/group,root/'validation'/group)
    for group,name in [('test','main'),('check_half_ms','half_ms'),('check_2ms','two_ms')]:
        if (source/group).exists():evaluations(source/group,root/'evaluation'/name)
    evaluations(source/'step_diagnosis',root/'step_diagnosis')
    with ZipFile(root/'step_diagnosis'/'traces.zip','w',ZIP_DEFLATED) as archive:
        for file in sorted((source/'step_diagnosis').glob('*.npz')):archive.write(file,file.name)
    for file in (source/'figures').iterdir():
        if file.is_file():copy(file,root/'figures'/file.name)
    for name in ['comparison.mp4','correct_throw.mp4','manifest.json','preview.png']:
        copy(source/'video'/name,root/'video'/name)
    with ZipFile(root/'video'/'raw_recordings.zip','w',ZIP_DEFLATED) as archive:
        for file in sorted((source/'video').glob('*__on__*/*')):
            if file.is_file():archive.write(file,str(file.relative_to(source/'video')))
    for name in ['tests.log','tests.xml']:
        copy(source/name,root/name)
    environment={'python':sys.version,'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in (
                     'torch','mujoco','numpy','gymnasium','trimesh','scipy','matplotlib','imageio')}}
    files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
    (root/'manifest.json').write_text(json.dumps({'environment':environment,'files':files},indent=2)+'\n')


if __name__=='__main__':main()
