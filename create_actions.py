import json

actions = {}
actions['c'] = {}
actions['c']['filters'] = [1, 8, 16, 32, 64]
actions['c']['kernel_size'] = [3, 5, 7]
actions['c']['strides'] = [1, 2, 3]
actions['c']['padding'] = ['SAME']
actions['c']['activation'] = ['relu']

actions['b'] = []
actions['mp'] = {}
actions['mp']['pool_size'] = [2, 5, 7]
actions['mp']['strides'] = [1, 2, 3]
actions['mp']['padding'] = ['SAME']

actions['ap'] = {}
actions['ap']['pool_size'] = [2, 5, 7]
actions['ap']['strides'] = [1, 2, 3]
actions['ap']['padding'] = ['SAME']

actions['d'] = {}
actions['d']['units'] = [25, 50, 75, 100]

actions['o'] = {}
actions['o']['units'] = [10]

actions['f'] = []
actions['lr'] = {}
actions['lr']['lr'] = [10**x for x in [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]]

with open('actions.json', 'w') as outfile:
    json.dump(actions, outfile)
