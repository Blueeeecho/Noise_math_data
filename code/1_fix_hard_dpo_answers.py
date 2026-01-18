import json
import os


def execute_python_code(code):
    local_namespace = {}
    exec(code, {}, local_namespace)
    if "solve" not in local_namespace:
        return None
    result = local_namespace["solve"]()
    return result


def extract_plan_block(content):
    start_tag = "<plan>"
    end_tag = "</plan>"
    start = content.find(start_tag)
    end = content.find(end_tag)
    if start == -1 or end == -1:
        return None
    start += len(start_tag)
    if start >= end:
        return None
    return content[start:end].strip()


def replace_answer(content, value):
    start_tag = "<answer>"
    end_tag = "</answer>"
    start = content.find(start_tag)
    end = content.find(end_tag, start + len(start_tag)) if start != -1 else -1
    if start == -1 or end == -1:
        return content
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    new_block = f"{start_tag}{text}{end_tag}"
    return content[:start] + new_block + content[end + len(end_tag) :]


def process_message(message):
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if not isinstance(content, str):
        return message
    code_block = extract_plan_block(content)
    if not code_block:
        return message
    try:
        result = execute_python_code(code_block)
    except Exception:
        return message
    if result is None:
        return message
    new_content = replace_answer(content, result)
    new_message = dict(message)
    new_message["content"] = new_content
    return new_message


def process_file(path):
    tmp_path = path + ".tmp"
    with open(path, "r", encoding="utf-8") as fin, open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            obj = json.loads(line_stripped)
            for key in ("chosen", "rejected"):
                messages = obj.get(key)
                if not isinstance(messages, list):
                    continue
                new_messages = []
                for message in messages:
                    new_messages.append(process_message(message))
                obj[key] = new_messages
            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")
    os.replace(tmp_path, path)


if __name__ == "__main__":
    dataset_path = (
        "/root/autodl-tmp/A-new/data/dpo/new_1.5b_instruct/hard_dpo_all.jsonl"
    )
    process_file(dataset_path)

