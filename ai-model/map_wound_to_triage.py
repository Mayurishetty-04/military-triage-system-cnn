import os
import shutil

SRC = "/Users/pranathi/Documents/military-triage-system-cnn/Wound_dataset"

DEST = "ai-model/visual_dataset"

mapping = {
    "Abrasions": "green",
    "Bruises": "green",
    "Ingrown_nails": "green",

    "Cut": "yellow",

    "Burns": "red",
    "Laceration": "red",

    "Stab_wound": "black"
}

for src_folder, triage in mapping.items():
    src_path = os.path.join(SRC, src_folder)
    dest_path = os.path.join(DEST, triage)

    os.makedirs(dest_path, exist_ok=True)

    for file in os.listdir(src_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(
                os.path.join(src_path, file),
                dest_path
            )

print("✅ Dataset organized successfully.")
