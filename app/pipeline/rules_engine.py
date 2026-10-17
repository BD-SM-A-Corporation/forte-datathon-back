from pathlib import Path
from typing import Dict, Optional, List
import yaml
from app.config import settings
from app.schemas import RuleItem


class RulesEngine:
    def __init__(self, rules_file: Optional[Path] = None):
        self.rules_file = rules_file or (settings.BASE_DIR / settings.RULES_PATH)
        self.rules_map: Dict[str, RuleItem] = {}
        self.load_rules()

    def load_rules(self):
        if not self.rules_file.exists():
            return
        with open(self.rules_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            for item in data.get("rules", []):
                rule = RuleItem(**item)
                self.rules_map[rule.event_type] = rule

    def get_rule(self, event_type: str) -> Optional[RuleItem]:
        return self.rules_map.get(event_type)

    def get_all_rules(self) -> List[RuleItem]:
        return list(self.rules_map.values())


rules_engine = RulesEngine()
