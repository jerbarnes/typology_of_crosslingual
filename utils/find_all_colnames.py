import glob
import pandas as pd
import os
from pathlib import Path

def run():
    files = glob.glob(r"{}/../*/**/[a-zA-Z]*.xlsx".format(str(Path(__file__).parent)), recursive=True)

    paths = []
    cols = []

    for file in files:
        df = pd.read_excel(file)
        cols.extend(df.columns)

        # Make path relative to repo root
        parts = Path(file).absolute().resolve().parts
        path = os.path.join(*parts[parts.index("typology_of_crosslingual") + 1:])
        paths.extend([path] * len(df.columns))

    pd.DataFrame({"path": paths, "colname": cols}).to_csv(Path(__file__).parent / "all_colnames.tsv", index=False, sep="\t")
    print("Updated utils/all_colnames.tsv")

if __name__ == "__main__":
    run()
