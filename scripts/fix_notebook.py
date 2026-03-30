import json

notebook_path = "/Users/forget/Desktop/Project_Momentum_AI/notebooks/new_exploration/Momentum_MA_Commodities_Fixed.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb_data = json.load(f)

# Loop over all cells and replace 'adjClose' with 'close' inside the source code
for cell in nb_data.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            new_line = line.replace("'adjClose'", "'close'")
            new_line = new_line.replace("\"adjClose\"", "\"close\"")
            new_source.append(new_line)
        cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=1)

print("Notebook updated successfully.")
