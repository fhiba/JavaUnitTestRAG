import json
import re
from dotenv import load_dotenv
from vector_store import get_vector_store

load_dotenv()

def main():
    with open("../datasets/dataset.json", "r") as f:
        data = json.load(f)

    index, index_name = get_vector_store()

    records = []
    for item in data:
        m = re.search(r"class\s+(\w+)", item["class"])
        doc_id = m.group(1) if m else str(data.index(item))

        text = item["class"] + "\n" + item.get("description", "")

        records.append({
            "id":       doc_id,
            "text":     text,
            "class":    item["class"],
            "test":     item.get("tests", ""),
            "description": item.get("description", "")
        })

    index.upsert_records(records=records, namespace="default")
    print(f"Upserted {len(records)} records into '{index_name}'")

if __name__ == "__main__":
    main()
