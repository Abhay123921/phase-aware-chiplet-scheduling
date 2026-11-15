from pathlib import Path
import re
import csv

root = Path("timeloop/outputs/resnet50")
rows = []

for stats_file in sorted(root.glob("*/timeloop-mapper.stats.txt")):
    content = stats_file.read_text(errors="ignore")

    section = re.search(
        r"=== buffer ===(.*?)(?==== DRAM ===)",
        content,
        re.S,
    )

    if not section:
        print("WARNING: Buffer section missing:", stats_file.parent.name)
        continue

    buffer_text = section.group(1)

    capacity_match = re.search(r"Effective size\s*:\s*(\d+)", buffer_text)
    tensor_values = {}

    for tensor in ("Weights", "Inputs", "Outputs"):
        match = re.search(
            rf"{tensor}:.*?Utilized capacity\s*:\s*(\d+)",
            buffer_text,
            re.S,
        )
        if match:
            tensor_values[tensor] = int(match.group(1))

    if not capacity_match or len(tensor_values) != 3:
        print("WARNING: Incomplete data:", stats_file.parent.name)
        continue

    weights = tensor_values["Weights"]
    inputs = tensor_values["Inputs"]
    outputs = tensor_values["Outputs"]
    capacity = int(capacity_match.group(1))

    rows.append({
        "layer_id": stats_file.parent.name,
        "buffer_capacity_words": capacity,
        "weights_words": weights,
        "inputs_words": inputs,
        "outputs_words": outputs,
        "total_occupied_words": weights + inputs + outputs,
    })

output = Path("results/layer_sram_reference.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

occupancies = [r["total_occupied_words"] for r in rows]

print("Saved:", output)
print("Layers extracted:", len(rows))
print("Maximum occupancy:", max(occupancies))
print("Minimum occupancy:", min(occupancies))
print("Average occupancy:", round(sum(occupancies) / len(occupancies), 2))
print("Layers exceeding capacity:", sum(
    r["total_occupied_words"] > r["buffer_capacity_words"] for r in rows
))
