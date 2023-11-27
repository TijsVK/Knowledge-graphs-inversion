import inspect
import pyrdf4j
from pathlib import Path
import shutil

template_folder = Path(inspect.getfile(pyrdf4j)).parent / 'repo_type_templates'
this_folder = Path(__file__).parent

# copy this/repo-config.tll to template_folder/graphdb.ttl
shutil.copyfile(this_folder / 'repo-config.ttl', template_folder / 'graphdb.ttl')