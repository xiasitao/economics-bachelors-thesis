import pytask
import pandas as pd

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

@pytask.mark.depends_on(BUILD_PATH / 'role_models/ses_scores.pkl')
@pytask.mark.produces(BUILD_PATH / 'latex/role_model_overview.tex')
def task_role_model_list(produces: Path):
    role_model_data = pd.read_pickle(BUILD_PATH / 'role_models/role_model_data.pkl')
    scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
    role_model_data = role_model_data.join(scores, how='inner')
    role_model_data = role_model_data.sort_index(key=lambda name: name.str.lower())

    latex = r"\begin{longtable}{lccccccc}\toprule role model & sex & main prof. & y.o.b. & \#low & \#high & $\in \mathcal{R}_\text{distinct}$ \\\toprule"
    for i in range(len(role_model_data)):
        role_model = role_model_data.iloc[i].fillna("")
        line = f'{role_model.name} & {"m" if role_model["sex"]==0.0 else "f"} & {role_model["profession"]} & {role_model["birth_year"]} & {role_model["low_ses_count"]} & {role_model["high_ses_count"]} & {"$XYZXYZnotin$" if role_model["low_ses"] and role_model["high_ses"] else "$XYZXYZin$"}'
        line = line.replace('XYZXYZ', '\\')
        latex += f'{line} \\\\\n'
    latex += r"\bottomrule\caption{Overview of role models. Abbreviations: \textit{nat.}: nationality, \textit{main prof.}: main profession, \textit{y.o.b.}: year of birth, \textit{\#low/\#high}: number of mentions by low-SES/high-SES study participants, $\in \mathcal{R}_\text{distinct}$: whether they are in the set of role models with distinct SES association.}\label{tab:role_model_overview}\end{longtable}"
    produces.write_text(latex)

if __name__ == '__main__':
    task_role_model_list(BUILD_PATH / 'latex/role_model_overview.tex')