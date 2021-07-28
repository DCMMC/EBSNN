import pickle

filename = './data/29_header_payload_all.traffic'

with open(filename, 'r') as f:
    traffic = f.readlines()


with open('./data/29_payload_all.traffic','w') as f:
    for i in range(len(traffic)):
        s_traffic = traffic[i].split()
        if s_traffic[10] == '11':
            payload = s_traffic[0] + ' ' + ' '.join(s_traffic[29:])
        else:
            payload = s_traffic[0] + ' ' + ' '.join(s_traffic[41:])
        f.write(payload + '\n')
