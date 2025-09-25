import json
from pathlib import Path
from typing import Any
from tqdm import tqdm

from coauthor_interface.thought_toolkit.utils import (
    custom_serializer,
    get_spacy_similarity,
)

from coauthor_interface.thought_toolkit.parser_all_levels import (
    SameSentenceMergeAnalyzer,
    parse_level_2_actions,
    parse_level_3_actions,
)

from coauthor_interface.thought_toolkit.active_plugins import ACTIVE_PLUGINS


def parse_level_1_actions(
    coauthor_logs_by_session: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Parse level 1 actions from the raw logs."""
    level_1_actions_per_session = {}

    for session in tqdm(coauthor_logs_by_session):
        actions_analyzer = SameSentenceMergeAnalyzer(
            last_action=None, raw_logs=coauthor_logs_by_session[session]
        )

        actions_lst, _ = actions_analyzer.parse_actions_from_logs(
            all_logs=coauthor_logs_by_session[session], last_action=None
        )

        level_1_actions_per_session[session] = actions_lst

    # Add level_1_action_type to each action
    for session_key, actions in level_1_actions_per_session.items():
        for action in actions:
            action["level_1_action_type"] = action["action_type"]

    return level_1_actions_per_session


def parse_level_2_actions_from_level_1(
    level_1_actions: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Parse level 2 actions from level 1 actions."""
    return parse_level_2_actions(level_1_actions, similarity_fcn=get_spacy_similarity)


def parse_level_3_actions_from_level_2(
    level_2_actions: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Parse level 3 actions from level 2 actions."""
    return parse_level_3_actions(level_2_actions, similarity_fcn=get_spacy_similarity)


def populate_priority_list(actions_dict: dict[str, list[dict[str, Any]]], level: str) -> list[str]:
    """Generate a list of unique action types from a specific level."""
    priority_set: set[str] = set()
    for _, actions in actions_dict.items():
        for action in actions:
            if level in action:
                priority_set.add(action[level])
    return list(priority_set)


def action_type_priority_sort(
    priority_list: list[str], actions_dict: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    """Sort and assign action types based on priority list."""
    for _, actions in actions_dict.items():
        for action in actions:
            # Iterate through the priority list and match against all levels
            for priority_action in priority_list:
                if (
                    action.get("level_1_action_type") == priority_action
                    or action.get("level_2_action_type") == priority_action
                    or action.get("level_3_action_type") == priority_action
                ):
                    action["action_type"] = priority_action
                    break
    return actions_dict


def process_logs(input_file: Path, output_dir: Path) -> None:
    """Process logs through all levels of analysis and save results."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input logs
    with open(input_file) as f:
        coauthor_logs_by_session = json.load(f)

    # Process through all levels
    level_1_actions = parse_level_1_actions(coauthor_logs_by_session)
    level_2_actions = parse_level_2_actions_from_level_1(level_1_actions)
    level_3_actions = parse_level_3_actions_from_level_2(level_2_actions)

    # Generate priority-based actions
    custom_priority_list = [plugin.get_plugin_name() for plugin in ACTIVE_PLUGINS]
    priority_actions = action_type_priority_sort(custom_priority_list, level_3_actions)

    # Save all results
    output_files = {
        "level_1_actions_per_session.json": level_1_actions,
        "level_2_actions_per_session.json": level_2_actions,
        "level_3_actions_per_session.json": level_3_actions,
        "action_type_with_priority_per_session.json": priority_actions,
    }

    for filename, data in output_files.items():
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, default=custom_serializer)


# After running the entire Python file, you will generate four different JSON files:
# 1. 'level_1_actions_per_session.json' - Contains the parsed level 1 actions organized by session.
# 2. 'level_2_actions_per_session.json' - Builds upon level 1 actions, adding semantic differences and coordination scores.
# 3. 'level_3_actions_per_session.json' - Adds advanced interpretations, such as topic shifts and mindless edits or echoes.
# 4. 'action_type_with_priority_per_session.json' - Applies priority-based sorting to action types for refined analysis.
if __name__ == "__main__":
    # Example usage
    script_dir = Path(__file__).parent
    input_file = script_dir / "raw_keylogs_for_analysis.json"
    output_dir = script_dir / "output"
    process_logs(input_file, output_dir)
