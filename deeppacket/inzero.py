
# with open('./data/raw29.traffic') as f:
with open('../data/29_header_payload_all.traffic') as f:
    data =f.readlines()

with open('./data/middle.traffic', 'w') as f:
    for d in data:
        d = d.strip().split()
        if len(d) < 1501:
            d += ['00'] * (1501 - len(d))
        else:
            d = d[:301]
        f.write(' '.join(d))
        f.write('\n')
