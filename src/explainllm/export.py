import json
from typing import Dict, Optional


def export_relevance_json(result: Dict, path: Optional[str] = None) -> str:
    """
    Serialize relevance results to JSON.

    Args:
        result: Dict from ``generate_with_relevance()``.
        path:   File path to write. If None, returns JSON string only.

    Returns:
        The JSON string.
    """
    serialisable = {
        "prompt": result["prompt"],
        "max_new_tokens": result.get("max_new_tokens"),
        "generated_text": result["generated_text"],
        "full_text": result["full_text"],
        "prompt_tokens": result["prompt_tokens"],
        "prompt_len": result["prompt_len"],
        "generated_tokens": result["generated_tokens"],
        "all_token_strings": result["all_token_strings"],
        "token_details": result["token_details"],
    }
    json_str = json.dumps(serialisable, indent=2)

    if path is not None:
        with open(path, "w") as f:
            f.write(json_str)
        print(f"Saved -> {path}")

    return json_str
