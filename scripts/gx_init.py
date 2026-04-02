import os
import great_expectations as gx
from yaml import dump

def init_gx():
    # 1. Create the base directory
    gx_dir = "great_expectations"
    if not os.path.exists(gx_dir):
        os.makedirs(gx_dir)
        print(f"Created {gx_dir} directory.")

    # 2. Initialize the context
    # Starting from GE 1.x, we use the new programmatic API
    context = gx.get_context(project_root_dir=".")
    print("GX Data Context initialized.")

    # 3. Create subfolders for suites and checkpoints if they don't exist
    for sub in ["expectation_suites", "checkpoints", "uncommitted"]:
        spath = os.path.join(gx_dir, sub)
        if not os.path.exists(spath):
            os.makedirs(spath)
            print(f"Created {spath} directory.")

if __name__ == "__main__":
    init_gx()
