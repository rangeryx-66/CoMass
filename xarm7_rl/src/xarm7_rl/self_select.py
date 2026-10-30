"""Select checkpoints using their own physical world only, before cross tests."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil

from .config import load_config
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--training-dir",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=30000)
    p.add_argument("--screen-episodes",type=int,default=10)
    p.add_argument("--episodes",type=int,default=50)
    a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    config=load_config(a.config);config['device']='cpu'
    candidates=sorted(Path(a.training_dir).glob('step_*.pt'))+[Path(a.training_dir)/'best.pt',Path(a.training_dir)/'last.pt']
    screened=[]
    for checkpoint in candidates:
        folder=root/'screen'/checkpoint.stem;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'episodes.jsonl').open('w') as log,redirect_stdout(log):
            result=evaluate(str(checkpoint),config,a.screen_episodes,str(folder),seed=a.seed)
        row={'checkpoint':str(checkpoint),'success_rate':result['success_rate'],'mean_error_m':result['mean_landing_error_m']}
        screened.append(row)
        (root/'screen.json').write_text(json.dumps(screened,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    def order(row):return (-row['success_rate'],row['mean_error_m'] if row['mean_error_m'] is not None else float('inf'))
    finalists=[]
    for row in sorted(screened,key=order)[:3]:
        checkpoint=Path(row['checkpoint']);folder=root/'validation'/checkpoint.stem;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'episodes.jsonl').open('w') as log,redirect_stdout(log):
            result=evaluate(str(checkpoint),config,a.episodes,str(folder),seed=a.seed)
        finalists.append({'checkpoint':str(checkpoint),'success_rate':result['success_rate'],'mean_error_m':result['mean_landing_error_m']})
    selected=sorted(finalists,key=order)[0]
    shutil.copyfile(selected['checkpoint'],root/'selected.pt')
    (root/'selection.json').write_text(json.dumps({'criterion':'Own-world success, then own-world landing error; no cross-world scores used','seed':a.seed,'episodes':a.episodes,'finalists':finalists,'selected':selected},indent=2)+'\n')
    print(json.dumps({'selected':selected}),flush=True)


if __name__=='__main__':main()
