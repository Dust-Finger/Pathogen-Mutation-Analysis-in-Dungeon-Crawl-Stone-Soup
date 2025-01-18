import re

class DCSSLogParser:
    def __init__(self, log_file):
        self.log_file = log_file

    def parse_log(self):
        """
        Parses the DCSS game log and extracts states and events.
        Returns a list of state-action-reward tuples.
        """
        events = []
        with open(self.log_file, 'r') as file:
            for line in file:
                parsed_event = self._parse_line(line)
                if parsed_event:
                    events.append(parsed_event)
        return events

    def _parse_line(self, line):
        """
        Parses a single line from the DCSS log file.
        Extracts relevant data like player state, action, and outcomes.
        """
        # Example pattern for parsing (adjust based on actual log format)
        state_pattern = r"Turn (\d+): HP:(\d+)/(\d+) MP:(\d+)/(\d+) Pos:\((\-?\d+), (\-?\d+)\)"
        action_pattern = r"Action: (.+)"
        reward_pattern = r"Reward: (\d+)"

        state_match = re.search(state_pattern, line)
        action_match = re.search(action_pattern, line)
        reward_match = re.search(reward_pattern, line)

        if state_match and action_match and reward_match:
            turn, hp, max_hp, mp, max_mp, x, y = map(int, state_match.groups())
            action = action_match.group(1)
            reward = int(reward_match.group(1))
            return {
                'turn': turn,
                'state': {
                    'hp': hp,
                    'max_hp': max_hp,
                    'mp': mp,
                    'max_mp': max_mp,
                    'position': (x, y),
                },
                'action': action,
                'reward': reward,
            }
        return None
