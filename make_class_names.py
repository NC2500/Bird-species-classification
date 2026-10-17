import sys

# Usage: python make_class_names.py train.txt
# Reads train.txt (lines look like: Black_footed_Albatross_0019_416160254.jpg 0)
# and writes class_names.txt with one bird name per line, in label order.
path = sys.argv[1] if len(sys.argv) > 1 else 'train.txt'

names = {}
for line in open(path):
    parts = line.split()
    if len(parts) != 2:
        continue                                   # skip blank or odd lines
    file_name, label = parts[0], int(parts[1])
    # Bird name = file name minus the last two parts (image number and id).
    names[label] = ' '.join(file_name.split('_')[:-2])

# Safety check: labels must be exactly 0 to 199.
assert sorted(names) == list(range(200)), 'Expected labels 0 to 199'

with open('class_names.txt', 'w') as f:
    f.write('\n'.join(names[i] for i in range(200)))

print('Wrote class_names.txt with', len(names), 'classes')
print('First:', names[0], '| Last:', names[199])
