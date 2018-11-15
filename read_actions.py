import json
import itertools

def get_actions(action_file):
    with open(action_file) as data_file:
        actions = json.load(data_file)

    all_actions = []

    for x in actions:
        if actions[x]:
            all_ = []
            for y in actions[x]:
                all_.append(actions[x][y])
            for comb in itertools.product(*all_):
                all_actions.append((x, dict(zip(actions[x].keys(), list(comb)))))
        else:
            all_actions.append((x, {}))
    return all_actions
