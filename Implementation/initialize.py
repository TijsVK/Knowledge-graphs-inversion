import inspect
from pathlib import Path
import sys
pyrdf4j_path = Path(__file__).parent / "pyrdf4j"
sys.path.append(str(pyrdf4j_path))
import pyrdf4j
import shutil

template_folder = Path(inspect.getfile(pyrdf4j)).parent / 'repo_type_templates'
this_folder = Path(__file__).parent

# copy this/repo-config.tll to template_folder/graphdb.ttl
shutil.copyfile(this_folder / 'repo-config.ttl', template_folder / 'graphdb.ttl')