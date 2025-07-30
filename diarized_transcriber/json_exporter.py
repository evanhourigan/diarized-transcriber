import json

def save_json(result, output_path):
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
