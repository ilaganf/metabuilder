import json

actions = {}
actions['C'] = {}
actions['C']['filters'] = [1, 8, 16, 32, 64]
actions['C']['kernel_size'] = [3, 5, 7]
actions['C']['strides'] = [1, 2, 3]
actions['C']['padding'] = ['SAME']
actions['C']['activation'] = ['relu']

actions['B'] = []
actions['mp'] = {}
actions['mp']['pool_size'] = [2, 5, 7]
actions['mp']['strides'] = [1, 2, 3]
actions['mp']['padding'] = ['SAME']

actions['ap'] = {}
actions['ap']['pool_size'] = [2, 5, 7]
actions['ap']['strides'] = [1, 2, 3]
actions['ap']['padding'] = ['SAME']

actions['D'] = []

with open('actions.json', 'w') as outfile:
    json.dump(actions, outfile)
